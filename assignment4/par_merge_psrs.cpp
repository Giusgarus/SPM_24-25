/*
 * ----------------------------------------------------------------------------
 *  File: par_merge_psrs.cpp
 *  Description:
 *      Hybrid MPI + FastFlow implementation that combines a task‑parallel
 *      bottom‑up mergesort (shared‑memory) with a global Parallel Sample
 *      Sort (PSRS) redistribution (distributed‑memory).
 *
 *      Each MPI rank:
 *          1. Collectively reads its slice of the binary dataset through MPI‑IO.
 *          2. Sorts the local slice with an adaptive FastFlow farm composed of
 *             one emitter thread and n‑1 worker threads.
 *          3. Cooperates in a PSRS exchange so that the final concatenation
 *             of per‑rank vectors is globally ordered by key.
 *
 *      The program supports variable‑length payloads and automatically packs
 *      and unpacks records into byte‑contiguous buffers to avoid alignment
 *      issues during MPI communication.
 *
 *  Build:
 *      mpicxx -std=c++17 -O3 -march=native -I<fastflow_include_dir> \
 *              par_merge_psrs.cpp -lff
 *
 *  Example run (4 MPI ranks, 8 threads each):
 *      mpirun -np 4 ./par_merge_psrs -s 100M -t 8 -r 256
 *
 *  Author: <Giusepep Gabriele Russo>
 *  Date:   2025-06-03
 * ----------------------------------------------------------------------------
 */
// =======================================================================
//  Hybrid MPI + FastFlow MergeSort  –  merge divide-and-conquer
// =======================================================================
#define FF_ALL
#include <ff/ff.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <climits>
#include <vector>
#include <mpi.h>

using namespace ff;

// -----------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------

/**
 * @brief Plain old data structure representing one record of the dataset.
 *
 * The key is used for ordering; the payload is a dynamically‑allocated
 * opaque byte buffer whose length is set at runtime (PAYLOAD_SIZE).
 * Ownership is manual: whoever allocates the payload must later free it.
 */
struct Record {
    unsigned long key; 
    char* payload;
};
static std::size_t PAYLOAD_SIZE;

// -----------------------------------------------------------------------
// Comparator and grain threshold (for recursive merge)
// -----------------------------------------------------------------------

/**
 * @brief Strict‑weak ordering functor for STL algorithms.
 */
static auto cmpRecord = [](const Record& a, const Record& b){
    return a.key < b.key;
};


// -----------------------------------------------------------------------
// Task & FastFlow nodes  (emitter + worker + feedback)
// -----------------------------------------------------------------------

/**
 * @brief Operation type carried by a Task.
 *
 * SORT  – the range [left,right) must be sorted.
 * MERGE – the range is already split into [left,mid) and [mid,right)
 *         which need to be merged in place.
 */
enum class Op { SORT, MERGE };

/**
 * @brief Work item exchanged between the emitter and worker nodes.
 *
 * Describes the memory interval to operate on and the algorithm step
 * (sorting or merging) to execute.
 */
struct Task {
    Record* base;
    std::size_t left, mid, right;   // mid==left  → SORT
    Op op;
};

/* ------------------------------------------------------------------ */
/*  SIMPLE Worker: performs either std::sort or std::inplace_merge    */
/* ------------------------------------------------------------------ */
/**
 * @brief FastFlow node that executes a Task on its dedicated thread.
 */
struct Worker : ff_node_t<Task> {
    Task* svc(Task* t) override {
        if (t->op == Op::SORT)
            std::sort(t->base + t->left, t->base + t->right, cmpRecord);
        else
            std::inplace_merge(t->base + t->left,
                               t->base + t->mid,
                               t->base + t->right, cmpRecord);
        return t;                 // pointer returned to the emitter
    }
};




/**
 * @brief Coordinator node that generates new tasks, performs part of the
 *        work itself, and dynamically shrinks the worker pool.
 *
 * Acts as the root of a breadth‑first divide‑and‑conquer traversal,
 * building successive merge levels until the array is fully sorted.
 */
class Emitter : public ff_monode_t<Task> {
public:
    Emitter(Record* d, std::size_t n, int tot_thr)
        : base(d), N(n), nthreads(tot_thr), alive(tot_thr-1) {}

    Task* svc(Task* completed) override {

        if (first) { 
            first = false; 
            create_first_level();           // Create the first level of tasks (initial partition)
            emit_current_level();           // Emit the first level of tasks to workers
        }

        if (myTask) {                       // If emitter has a task to execute
            execute_my();                   // Execute its own task
            delete myTask;                  // Free memory
            myTask = nullptr;
        }

        // 'completed' counts only returns from workers
        if (completed && completed != EOS) {
            delete completed;               // Free completed task
            ++done;                         // Increment number of completed tasks
        }

        if (done < expected) return GO_ON;  // Wait for all tasks in current level

        // Level finished
        build_next_level();                 // Build next merge level
        emit_current_level();               // Emit next level tasks

        // If only one block remains and no more tasks to emit
        if (blocks.size() == 1 && expected == 0){
            broadcast_task(EOS);            // Signal end-of-stream to workers
            execute_my();                   // Final merge by emitter itself
            delete myTask;
            myTask = nullptr;
            return EOS;                // Emitter terminates
        }

        return GO_ON;
    }


private:

    struct MergeBlock {
        size_t left;
        size_t mid;
        size_t right;
    };

    void create_first_level() {
        std::size_t blk = (N + nthreads - 1) / nthreads; // Compute block size per thread
        for (int w = 0; w < nthreads; ++w) {
            std::size_t L = w * blk;
            if (L >= N) break;
            std::size_t R = std::min(N, L + blk);
            blocks.push_back({L, L, R});  // mid=L means sort operation
        }
    }

    void emit_current_level() {
        queue.clear();
        Op op = (blocks.size() == static_cast<size_t>(nthreads)) ? Op::SORT : Op::MERGE;

        for (auto b : blocks)
            queue.push_back(new Task{base, b.left, b.mid, b.right, op}); // Create tasks for each block

        // The emitter always takes the last task, if any
        if (!queue.empty()) {
            myTask = queue.back();
            queue.pop_back();
        } else {
            myTask = nullptr;               // Defensive: no task for emitter
        }

        // 'expected' is the number of tasks sent to workers only
        expected = queue.size();

        // Send all tasks in the queue to workers
        while (!queue.empty()) {
            ff_send_out(queue.front());
            queue.pop_front();
        }
    }

    void build_next_level() {
        std::vector<MergeBlock> new_blocks;
        for (size_t i = 0; i < blocks.size(); i += 2) {
            if (i + 1 == blocks.size()) {
                new_blocks.push_back(blocks[i]); // Odd block out, carry forward
            } else {
                size_t L = blocks[i].left;
                size_t M = blocks[i].right;
                size_t R = blocks[i + 1].right;
                new_blocks.push_back({L, M, R}); // Merge two adjacent blocks
            }
        }
        blocks = new_blocks;
        done = 0; // Reset completed count for new level
    }

    void execute_my() {
        if (myTask->op == Op::SORT)
            std::sort(base + myTask->left, base + myTask->right, cmpRecord); // Local sort
        else
            std::inplace_merge(base + myTask->left, base + myTask->mid, base + myTask->right, cmpRecord); // Local merge
    }

    // void shrink_workers() {
    //     std::size_t need = expected - 1; 
    //     int surplus = alive - static_cast<int>(need);
    //     for (int i = 0; i < surplus ; i++) {
    //         ff_send_out(EOS);
    //         --alive;
    //     }
    // }

    /* Data members */
    Record* base; 
    std::size_t N; 
    int nthreads, alive;
    bool first{true};

    std::deque<Task*> queue;                // Queue of tasks to be sent to workers
    std::vector<MergeBlock> blocks;         // Current partitioning of the array
    std::size_t done{0}, expected{0};       // Counters for completed and expected tasks
    Task* myTask{nullptr};                  // Task executed by the emitter itself
};


/**
 * @brief Launch a FastFlow farm to sort a vector of Record objects.
 *
 * @param vec       Vector to be sorted.
 * @param nthreads  Total number of threads (>=1). If 1, the call
 *                  downgrades to sequential std::sort.
 */
void mergesort_fastflow(std::vector<Record>& vec, int nthreads) {

    /* --- sequential case -------------------------------------- */
    if (nthreads <= 1) {
        std::sort(vec.begin(), vec.end(), cmpRecord);
        return;
    }

    /* --- parallel case (>=2 threads: emitter + worker) ---------- */
    std::vector<ff_node*> workers;
    for (int i = 0; i < nthreads - 1; ++i)
        workers.push_back(new Worker);

    auto* emitter = new Emitter(vec.data(), vec.size(), nthreads);

    ff_farm farm(std::move(workers));
    farm.add_emitter(emitter);
    farm.remove_collector();
    farm.wrap_around();

    if (farm.run_and_wait_end() != 0)
        std::cerr << "FastFlow error\n";
}



std::size_t parse_size(std::string s){
    std::size_t m=1; char c=s.back();
    if(c=='K'||c=='k') m=1'000ULL;
    else if(c=='M'||c=='m') m=1'000'000ULL;
    else if(c=='G'||c=='g') m=1'000'000'000ULL;
    if(!std::isdigit(c)) s.pop_back();
    return std::stoull(s)*m;
}

/**
 * @brief Collective MPI‑IO loader that reads a contiguous slice assigned
 *        to the calling rank.
 *
 * @param fname      Path to the binary dataset.
 * @param first_idx  Global index of the first record owned by this rank.
 * @param count      Number of records to read.
 *
 * The function packs raw bytes into a temporary buffer and then rebuilds
 * the vector of Record objects locally.
 */
std::vector<Record>
load_records_mpi(const std::string& fname,
                 std::size_t        first_idx,
                 std::size_t        count)
{
    std::vector<Record> v(count);
    if (count == 0) return v;

    /* open file collectively */
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, fname.c_str(),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    constexpr MPI_Offset HEADER = 16;              /* 8‑byte N + 4‑byte P + 4‑byte pad */
    const std::size_t    REC_SZ = 8 + PAYLOAD_SIZE;/* key + payload */

    /* offset to my first record */
    MPI_Offset offset = HEADER + static_cast<MPI_Offset>(first_idx) * REC_SZ;

    /* temp buffer (contiguous byte blob) */
    std::vector<unsigned char> buf(REC_SZ * count);

    /* collective, contiguous read */
    MPI_File_read_at_all(fh, offset,
                         buf.data(),
                         buf.size(),
                         MPI_BYTE,
                         MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    /* unpack into Record vector */
    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(&v[i].key, buf.data() + i * REC_SZ, 8);
        v[i].payload = new char[PAYLOAD_SIZE];
        std::memcpy(v[i].payload,
                    buf.data() + i * REC_SZ + 8,
                    PAYLOAD_SIZE);
    }
    return v;
}




/**
 * @brief Writes the globally sorted records to a binary file using MPI-IO.
 *
 * Each MPI rank writes its own contiguous block of sorted records at the correct
 * offset in the output file. The file header (total number of records, payload size,
 * and padding) is written only by rank 0. All ranks synchronize to ensure the header
 * is written before any data blocks.
 *
 * @param fname   Output file name.
 * @param recs    Vector of sorted records to write (local to this rank).
 * @param counts  Number of records per rank (global partitioning).
 * @param displs  Displacement (starting index) per rank.
 * @param rank    MPI rank of the calling process.
 * @param nprocs  Total number of MPI processes.
 */
void write_sorted_records_mpi(const std::string& fname,
                              const std::vector<Record>& recs,
                              const std::vector<int>& counts,
                              const std::vector<int>& displs,
                              int rank, int nprocs, uint64_t N,
                              uint32_t P,
                              uint32_t pad) {
    const std::size_t REC_SZ = 8 + PAYLOAD_SIZE; // 8 bytes for key + payload size

    // Open the output file collectively for writing
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, fname.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    // Only rank 0 writes the file header (total records, payload size, padding)
    if (rank == 0) {
        // uint64_t N = std::accumulate(counts.begin(), counts.end(), 0ULL);
        // uint32_t P = static_cast<uint32_t>(PAYLOAD_SIZE);
        // uint32_t pad = 0;

        // Write N at offset 0, P at offset 8, pad at offset 12
        MPI_File_write(fh,
            &N, 1, MPI_UINT64_T, MPI_STATUS_IGNORE);
        MPI_File_write(fh,
            &P, 1, MPI_UINT32_T, MPI_STATUS_IGNORE);
        MPI_File_write(fh,
            &pad, 1, MPI_UINT32_T, MPI_STATUS_IGNORE);
    }

    // Synchronize all ranks to ensure header is written before data
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute the file offset for this rank's data (skip 16-byte header)
    MPI_Offset offset = 16 + static_cast<MPI_Offset>(displs[rank]) * REC_SZ;

    // Pack local records into a contiguous byte buffer
    std::vector<unsigned char> buffer(REC_SZ * recs.size());
    for (std::size_t i = 0; i < recs.size(); ++i) {
        // Copy key (8 bytes)
        std::memcpy(buffer.data() + i * REC_SZ, &recs[i].key, 8);
        // Copy payload (PAYLOAD_SIZE bytes)
        std::memcpy(buffer.data() + i * REC_SZ + 8, recs[i].payload, PAYLOAD_SIZE);
    }


    auto write_large = [&](MPI_Offset off,
                           const unsigned char* ptr,
                           MPI_Offset nbytes){
        const MPI_Offset CHUNK = static_cast<MPI_Offset>(INT_MAX);
        MPI_Offset done = 0;
        while (done < nbytes) {
             int chunk = static_cast<int>(std::min(CHUNK, nbytes - done));
             MPI_File_write_at(fh, off + done,
                               const_cast<unsigned char*>(ptr + done),
                               chunk,            /* count (<= INT_MAX) */
                               MPI_BYTE,
                               MPI_STATUS_IGNORE);
             done += chunk;
         }
    };
 
    /* Scrivi il buffer completo (qualunque dimensione) */
    write_large(offset, buffer.data(),
                 static_cast<MPI_Offset>(buffer.size()));

    // Close the file collectively
    MPI_File_close(&fh);
}





/**
 * @brief Distributed Parallel Sample Sort (PSRS) that redistributes
 *        complete Record objects so that each rank receives its final
 *        globally‑ordered partition.
 *
 * The routine follows the classic PSRS pipeline:
 *   1. Local sampling
 *   2. Gathering of samples on root
 *   3. Broadcast of splitters
 *   4. Counting & packing into buckets
 *   5. All‑to‑all exchange
 *   6. Local final sort
 */
std::vector<Record>
psrs_distribute_records(const std::vector<Record>& local_rec,
                        int rank, int nprocs, int threads)
{
    /* Special case p==1: deep‑copy to avoid double‑free on payload ---------- */
    if (nprocs == 1) {
        std::vector<Record> copy(local_rec.size());
        for (std::size_t i = 0; i < local_rec.size(); ++i) {
            copy[i].key     = local_rec[i].key;
            copy[i].payload = new char[PAYLOAD_SIZE];
            std::memcpy(copy[i].payload, local_rec[i].payload, PAYLOAD_SIZE);
        }
        std::sort(copy.begin(), copy.end(), cmpRecord);
        return copy;
    }

    const int p       = nprocs;
    const int s       = p - 1;                     // oversample
    const std::size_t REC_SZ = 8 + PAYLOAD_SIZE;   // key + payload bytes

    /* datatype for a raw Record blob (extent = REC_SZ) */
    MPI_Datatype REC_TYPE;
    MPI_Type_contiguous(static_cast<int>(REC_SZ), MPI_BYTE, &REC_TYPE);
    MPI_Type_commit(&REC_TYPE);

    /* ---------- 1. local sampling on keys ------------------- */
    std::vector<unsigned long> samples(s);
    for (int i = 0; i < s; ++i) {
        if (local_rec.empty()) { samples[i] = 0; continue; }
        std::size_t idx = (i + 1) * local_rec.size() / (p + 1);
        if (idx >= local_rec.size()) idx = local_rec.size() - 1;
        samples[i] = local_rec[idx].key;
    }

    /* ---------- 2. gather samples on root -------------------- */
    std::vector<unsigned long> all_samples;
    if (rank == 0) 
        all_samples.resize(s * p);
    MPI_Gather(samples.data(), s, MPI_UNSIGNED_LONG,
               all_samples.data(), s, MPI_UNSIGNED_LONG,
               0, MPI_COMM_WORLD);

    /* ---------- 3. splitter selection and broadcast -------------------- */
    std::vector<unsigned long> splitters(p - 1);
    if (rank == 0) {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 1; i < p; ++i)
            splitters[i - 1] = all_samples[i * s];
    }
    MPI_Bcast(splitters.data(), p - 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    /* ---------- 4. count elements per bucket ------------------ */
    std::vector<int> sendcnt(p, 0);
    {
        int b = 0;
        for (const Record& r : local_rec) {
            while (b < p - 1 && r.key > splitters[b]) ++b;
            ++sendcnt[b];
        }
    }
    /* displacement in #record */
    std::vector<int> sdispls(p, 0);
    for (int i = 1; i < p; ++i) sdispls[i] = sdispls[i-1] + sendcnt[i-1];

    int total_send = sdispls.back() + sendcnt.back();

    /* ---------- 5. pack <key,payload> ---------------------- */
    std::vector<unsigned char> sendbuf(static_cast<std::size_t>(total_send) * REC_SZ);
    std::vector<int> write_idx = sdispls;          // current position (#record) per bucket
    for (const Record& r : local_rec) {
        int b = 0;
        while (b < p - 1 && r.key > splitters[b]) ++b;
        std::size_t off = static_cast<std::size_t>(write_idx[b]) * REC_SZ;
        std::memcpy(sendbuf.data() + off, &r.key, 8);
        std::memcpy(sendbuf.data() + off + 8, r.payload, PAYLOAD_SIZE);
        ++write_idx[b];
    }

    /* -------------- displacement & counts in *records* ---------- */
    /* (avoid int overflow on byte counts > 2 GB)                  */
    std::vector<int> recv_cnt(p);
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT,
                 recv_cnt.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> rdispls(p, 0);
    for (int i = 1; i < p; ++i) rdispls[i] = rdispls[i-1] + recv_cnt[i-1];

    int total_recv = rdispls.back() + recv_cnt.back();

    std::vector<unsigned char> recvbuf(static_cast<std::size_t>(total_recv) * REC_SZ);

    /* -------------- exchange per-record, not per-byte ----------- */
    MPI_Alltoallv(sendbuf.data(),
                  sendcnt.data(),  sdispls.data(), REC_TYPE,
                  recvbuf.data(),
                  recv_cnt.data(), rdispls.data(), REC_TYPE,
                  MPI_COMM_WORLD);

    /* rebuild the Record vector ----------------------- */
    std::vector<Record> result(total_recv);
    for (int i = 0; i < total_recv; ++i) {
        const unsigned char* src = recvbuf.data() + static_cast<std::size_t>(i) * REC_SZ;
        std::memcpy(&result[i].key, src, 8);
        result[i].payload = new char[PAYLOAD_SIZE];
        std::memcpy(result[i].payload, src + 8, PAYLOAD_SIZE);
    }

    /* final local sort -------------------------------- */
    mergesort_fastflow(result, threads);
    MPI_Type_free(&REC_TYPE);
    return result;
}


// =======================================================================
/**
 * @brief Program entry point: parses CLI options, orchestrates I/O,
 *        local sorting, PSRS redistribution, and timing.
 */
int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::cerr << "Error: MPI does not provide required threading level (FUNNELED)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::size_t N = 0; int threads = 1; PAYLOAD_SIZE = 8;
    std::string s_arg, r_arg;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            s_arg = argv[++i];                 // save literal (e.g., 100M)
            N = parse_size(s_arg);
        }
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            r_arg = argv[++i];                 // save literal (e.g., 64, 256K)
            PAYLOAD_SIZE = parse_size(r_arg);
        }
    }
    if (!N) {
        std::cerr << "Usage: " << argv[0] << " -s <size> [-t T] [-r B]\n";
        return 1;
    }

    // Build dataset filename
    std::string fname = "./Dataset/data_" + s_arg + "_" + r_arg + ".bin";

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Partition the dataset among ranks and compute offsets
    std::vector<int> counts(nprocs), displs(nprocs);
    int base = N / nprocs, rem = N % nprocs, off = 0;
    for (int i = 0; i < nprocs; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = off;
        off += counts[i];
    }
    int local_N = counts[rank];
    std::size_t first_idx = displs[rank];

    // Collective MPI-IO read of local records
    auto localRecs = load_records_mpi(fname, first_idx, local_N);

    // Synchronize and start sort timer
    MPI_Barrier(MPI_COMM_WORLD);
    double t_mid = MPI_Wtime();

    // Local sort using FastFlow mergesort
    mergesort_fastflow(localRecs, threads);

    // PSRS redistribution if more than one process
    std::vector<Record> finalRecs;
    if (nprocs == 1) {
        // Only one rank: already globally sorted
        finalRecs = std::move(localRecs); // zero-copy move
    } else {
        finalRecs = psrs_distribute_records(localRecs, rank, nprocs, threads);
    }

    // Stop timer and compute makespan via MAX reduction
    double t_mid2 = MPI_Wtime();


    // Free any remaining payloads in localRecs
    for (auto& r : localRecs) delete[] r.payload;
    localRecs.clear(); localRecs.shrink_to_fit();

    std::vector<int> counts_sorted(nprocs);
    int my_new_count = static_cast<int>(finalRecs.size());
    MPI_Allgather(&my_new_count, 1, MPI_INT,
                 counts_sorted.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
    std::vector<int> displs_sorted(nprocs, 0);
    for (int i = 1; i < nprocs; ++i)
        displs_sorted[i] = displs_sorted[i-1] + counts_sorted[i-1];



    // ==================== [TEST CODE BEGIN] ====================
    // Check if local block is sorted
    bool ok_local = std::is_sorted(finalRecs.begin(), finalRecs.end(), cmpRecord);
    if (!ok_local)
        std::cerr << "[rank " << rank << "] block is not sorted!\n";
    // ==================== [TEST CODE END] ======================

    // Write globally sorted records to new output file
    std::string outname = "./Dataset/sorted_" + s_arg + "_" + r_arg + ".bin";


    // Synchronize to ensure header is written before data
    //write_sorted_records_mpi(outname, finalRecs, counts, displs, rank, nprocs, N, static_cast<uint32_t>(PAYLOAD_SIZE), 0);
    write_sorted_records_mpi(outname, finalRecs,
                             counts_sorted, displs_sorted,
                             rank, nprocs, N,
                             static_cast<uint32_t>(PAYLOAD_SIZE), 0);

    double t_local = t_mid2 - t_mid;
    double t_local_get_data = t_mid - t_start;
    double t_local_write_data = MPI_Wtime() - t_mid2;

    MPI_Barrier(MPI_COMM_WORLD);


    // ==================== [TEST CODE BEGIN] ====================
    double t_global;
    MPI_Reduce(&t_local, &t_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double t_global_get_data;
    MPI_Reduce(&t_local_get_data, &t_global_get_data, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double t_global_write_data;
    MPI_Reduce(&t_local_write_data, &t_global_write_data, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double total_time = t_global_get_data + t_global + t_global_write_data;

    if (rank == 0) {
        std::cout << "Total (get data) time: " << t_global_get_data << " s\n";
        std::cout << "Total (sort) time: " << t_global << " s\n";
        std::cout << "Total (write) time: " << t_global_write_data << " s\n";
        std::cout << "TOTAL TIME: " << total_time << " s\n";
    }
    // ==================== [TEST CODE END] ====================


    // Dealloc memory and finalize MPI
    for (auto& r : finalRecs) delete[] r.payload;
    MPI_Finalize();
    return 0;
}