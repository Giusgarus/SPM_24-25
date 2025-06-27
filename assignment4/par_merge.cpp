/*
 * ----------------------------------------------------------------------------
 *  File: par_merge.cpp
 *  Description:
 *      MPI program that uses a hybrid scheme:
 *        • A FastFlow farm (emitter + workers) performs a parallel sort of
 *          the key array local to each rank.
 *        • A binary tree merge (MPI point‑to‑point) gathers a single,
 *          globally‑sorted sequence of keys on rank 0.
 *        • Rank 0 finally rebuilds the vector of full Record objects by
 *          matching the sorted keys to their payloads.
 *
 *      Build:
 *          mpicxx -std=c++17 -O3 -march=native -I<fastflow_include_dir>  \\
 *                 par_merge.cpp -lff
 *
 *      Example run (8 MPI ranks, 4 threads each):
 *          mpirun -np 8 ./par_merge -s 500M -t 4 -r 256
 *
 *  Author: <Giuseppe Gabriele Russo>
 *  Date:   2025‑06‑03
 * ----------------------------------------------------------------------------
 */
// =======================================================================
//  par_merge_seq.cpp  –  FastFlow sort
// =======================================================================
#define FF_ALL
#include <ff/ff.hpp>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace ff;

// ---------------------------------------------------------------------
// Basic data structures
// ---------------------------------------------------------------------
// Record payload size
#ifndef RPAYLOAD
#define RPAYLOAD 64 // default fallback
#endif

/**
 * @brief Plain old data structure representing one record.
 *
 * Contains the 64‑bit key used for ordering and a statically‑sized
 * payload buffer (RPAYLOAD bytes).
 */
struct Record {
    unsigned long key;
    char rpayload[RPAYLOAD];
};
static std::size_t PAYLOAD_SIZE = 64;
double t_start, t_mid, t_mid2 ,t_end, t_total;

using SortTask = std::pair<unsigned long *, unsigned long *>; // [begin,end)


// ---------------------------------------------------------------------
// Worker: sorts the block in‑place
// ---------------------------------------------------------------------
/**
 * @brief FastFlow worker node that performs an in‑place std::sort on the
 *        sub‑range described by a SortTask.
 */
struct Worker : ff_node_t<SortTask>
{
    SortTask *svc(SortTask *t) override
    {
        std::sort(t->first, t->second);
        return t; // feedback not used
    }
};

// ---------------------------------------------------------------------
// Emitter: creates the blocks, sends them, sorts its own, then EOS
// ---------------------------------------------------------------------
/**
 * @brief FastFlow emitter node that partitions the input array into
 *        chunks, dispatches them to workers, sorts its own chunk,
 *        and finally emits EOS.
 */
/* ------------------------------------------------------------------
 *  Emitter  (nuova versione “safe EOS”)
 * ------------------------------------------------------------------ */
class Emitter : public ff_monode_t<SortTask>
{
public:
    Emitter(unsigned long* base, std::size_t N, int T)
        : base(base), N(N), nthreads(T) {}

    SortTask* svc(SortTask* completed) override
    {
        /* ---------- feedback from worker ------------------------- */
        if (completed)                 // return pointer
            ++done;                      // one block finished

        /* ---------- first call: create & send blocks ------------- */
        if (!built) {
            build_tasks();               // fill 'tasks'
            built = true;
        }
        if (idx < tasks.size())          // off-load while there is work
            ff_send_out(&tasks[idx++]);

        /* ---------- sort its own block only once ----------------- */
        if (!sortedMine) {
            std::sort(myTask.first, myTask.second);
            sortedMine = true;
        }

        /* ---------- have all workers finished?  ------------------ */
        if (done == total_worker_tasks) {
            broadcast_task(EOS);         // now it's safe to terminate
            return EOS;                  // end of stream
        }
        return GO_ON;
    }

    SortTask my_block() const { return myTask; }

private:
    void build_tasks()
    {
        std::size_t blk = (N + nthreads - 1) / nthreads;
        for (int w = 0; w < nthreads; ++w) {
            std::size_t L = w * blk;
            if (L >= N) break;
            std::size_t R = std::min(N, L + blk);
            tasks.emplace_back(base + L, base + R);
        }
        myTask = tasks.back();           // the last chunk stays with the emitter
        tasks.pop_back();                // only worker blocks remain
        total_worker_tasks = tasks.size();
    }

    /* data */
    unsigned long*            base;
    std::size_t               N;
    int                       nthreads;
    std::vector<SortTask>     tasks;          // blocks for the workers
    SortTask                  myTask;         // emitter's block
    std::size_t               idx{0};         // next to send
    std::size_t               done{0};        // completed worker blocks
    std::size_t               total_worker_tasks{0};
    bool                      built{false};
    bool                      sortedMine{false};
};

// ---------------------------------------------------------------------
// fastflow_sort_records  (parallel sort; merge seq.)
// ---------------------------------------------------------------------
/**
 * @brief Sort a vector of 64‑bit keys using a FastFlow farm followed by a
 *        sequential k‑way merge.
 *
 * @param keys      Vector of keys to sort (modified in place).
 * @param nthreads  Total number of FastFlow threads (≥1).
 * @return          New vector containing the sorted keys.
 */
std::vector<unsigned long>
fastflow_sort_keys(std::vector<unsigned long> &keys, int nthreads)
{
    const std::size_t N = keys.size();
    if (nthreads <= 1)
    { // sequential case
        std::sort(keys.begin(), keys.end());
        return keys;
    }

    /* worker pool */
    std::vector<ff_node *> workers;
    for (int i = 0; i < nthreads - 1; ++i)
        workers.push_back(new Worker);

    auto *emitter = new Emitter(keys.data(), N, nthreads);
    ff_farm farm(std::move(workers));
    farm.add_emitter(emitter);
    farm.remove_collector(); // no collector
    /* NO wrap_around() */   // important
    if (farm.run_and_wait_end() != 0)
        std::cerr << "FastFlow error\n";

    /* sorted blocks: workers (nthreads‑1) + emitter block */
    std::vector<SortTask> blocks;
    blocks.reserve(nthreads);
    std::size_t blk = (N + nthreads - 1) / nthreads;
    for (int w = 0; w < nthreads - 1; ++w)
    {
        std::size_t L = w * blk;
        std::size_t R = std::min(N, L + blk);
        blocks.emplace_back(keys.data() + L, keys.data() + R);
    }
    blocks.push_back(emitter->my_block());

    /* sequential k‑way merge */
    std::vector<unsigned long> out;
    out.reserve(N);
    std::vector<std::size_t> idx(blocks.size(), 0);
    while (out.size() < N)
    {
        int best = -1;
        unsigned long minKey = ~0ULL;
        for (std::size_t b = 0; b < blocks.size(); ++b)
        {
            if (blocks[b].first + idx[b] == blocks[b].second)
                continue;
            unsigned long k = *(blocks[b].first + idx[b]);
            if (best == -1 || k < minKey)
            {
                best = b;
                minKey = k;
            }
        }
        out.push_back(*(blocks[best].first + idx[best]));
        ++idx[best];
    }

    assert(std::is_sorted(out.begin(), out.end()));

    return out;
}

// ---------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------
/**
 * @brief Load the entire binary dataset from disk (rank 0 only).
 *
 * File layout:
 *   uint64_t N   – number of records
 *   uint32_t P   – payload size in bytes
 *   uint32_t pad – ignored
 *   N × (uint64_t key + P‑byte payload)
 */
std::vector<Record> load_records(const std::string &f)
{
    std::ifstream in(f, std::ios::binary);
    if (!in)
        throw std::runtime_error("open file");
    uint64_t N;
    uint32_t P, pad;
    in.read((char *)&N, 8);
    in.read((char *)&P, 4);
    in.read((char *)&pad, 4);
    PAYLOAD_SIZE = P;
    std::vector<Record> v(N);
    for (uint64_t i = 0; i < N; ++i)
    {
        in.read((char *)&v[i].key, 8);
        in.read(v[i].rpayload, P);
    }
    return v;
}
std::size_t parse_size(std::string s)
{
    std::size_t m = 1;
    char c = s.back();
    if (c == 'K' || c == 'k')
        m = 1'000;
    else if (c == 'M' || c == 'm')
        m = 1'000'000;
    else if (c == 'G' || c == 'g')
        m = 1'000'000'000;
    if (!std::isdigit(c))
        s.pop_back();
    return std::stoull(s) * m;
}


/**
 * @brief Rebuild the ordered vector of Record objects given the global
 *        sequence of sorted keys (rank 0 only).
 *
 * @param sorted_keys  Keys in ascending order.
 * @param all_records  Unordered Record vector.
 * @return             Record vector sorted by key.
 */
std::vector<Record> rebuild_sorted_records(
    const std::vector<unsigned long>& sorted_keys,
    const std::vector<Record>& all_records)
{
    std::unordered_map<unsigned long, Record> key_to_record;
    for (const auto& r : all_records)
        key_to_record[r.key] = r;

    std::vector<Record> result;
    result.reserve(sorted_keys.size());
    for (auto k : sorted_keys)
        result.push_back(key_to_record[k]);

    t_end = MPI_Wtime();
    return result;
}



/**
 * @brief Writes a sequence of sorted records to a binary file.
 *
 * This function takes a vector of already sorted records and writes them to an output file
 * in binary format. The resulting file contains a header with the number of records, the payload
 * size of each record, and a padding field (currently set to zero), followed by the data
 * of each record (key and payload).
 *
 * @param outname Name of the output file to write the records to.
 * @param sorted Vector of sorted records to be written to the file.
 * @param payload_size Size, in bytes, of the payload of each record.
 *
 * @throws std::runtime_error If the output file cannot be opened.
 *
 * The produced file format is as follows:
 * - uint64_t: number of records (N)
 * - uint32_t: payload size (P)
 * - uint32_t: padding (currently 0)
 * - For each record:
 *     - uint64_t: record key
 *     - char[P]: record payload
 */
void write_sorted_records(const std::string& outname,
                          const std::vector<Record>& sorted,
                          std::size_t payload_size) {
    std::ofstream out(outname, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output file");

    uint64_t N = sorted.size();
    uint32_t P = payload_size;
    uint32_t pad = 0;

    // Write header
    out.write(reinterpret_cast<char*>(&N), sizeof(uint64_t));
    out.write(reinterpret_cast<char*>(&P), sizeof(uint32_t));
    out.write(reinterpret_cast<char*>(&pad), sizeof(uint32_t));

    // Write records
    for (const auto& r : sorted) {
        out.write(reinterpret_cast<const char*>(&r.key), sizeof(uint64_t));
        out.write(r.rpayload, P);
    }
}




/**
 * @brief Perform an MPI binary tree merge of per‑rank sorted key arrays.
 *
 * Each non‑root rank sends its keys up the tree; rank 0 ends up with the
 * globally sorted sequence.
 *
 * @param local_keys  Locally sorted keys (input/output).
 * @param rank        Caller’s MPI rank.
 * @param size        Total number of MPI ranks.
 * @return            Final merged keys for ranks that remain in the tree.
 */
std::vector<unsigned long> tree_merge_keys(std::vector<unsigned long>& local_keys, int rank, int size)
{
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                // Receive size of incoming block
                MPI_Status status;
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);

                // Receive the block of keys
                std::vector<unsigned long> recv_block(recv_size);
                MPI_Recv(recv_block.data(), recv_size, MPI_UNSIGNED_LONG, rank + step, 1, MPI_COMM_WORLD, &status);

                // Merge local keys with received block
                std::vector<unsigned long> merged;
                std::merge(local_keys.begin(), local_keys.end(),
                           recv_block.begin(), recv_block.end(),
                           std::back_inserter(merged));
                local_keys = std::move(merged);
            }
        } else {
            int dest = rank - step;

            // Send the size of the local_keys vector
            int send_size = local_keys.size();
            MPI_Send(&send_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

            // Send the local_keys block
            MPI_Send(local_keys.data(), send_size, MPI_UNSIGNED_LONG, dest, 1, MPI_COMM_WORLD);

            break;
        }
        step *= 2;
    }
    return local_keys;
}

/**
 * @brief Program entry point: parses CLI options, orchestrates parallel
 *        sort and merge, prints timing, and validates the result.
 */
int main(int argc, char *argv[])
{
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::cerr << "Error: MPI does not provide required threading level (FUNNELED)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------------------- parse CLI -------------------- */
    std::size_t N = 0; int threads = 1;
    std::string s_arg, r_arg;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) { s_arg = argv[++i]; N = parse_size(s_arg); }
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) { r_arg = argv[++i]; PAYLOAD_SIZE = parse_size(r_arg); }
    }
    if (!N) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " -s <size> [-t T] [-r B]\n";
        MPI_Finalize();
        return 1;
    }

    /* -------------------- single rank? ------------------ */
    if (size == 1) {
        /* timer start */
        t_start = MPI_Wtime();

        /* load the entire dataset */
        std::vector<Record> recs = load_records("./Dataset/data_" + s_arg + "_" + r_arg + ".bin");

        /* extract the keys */
        std::vector<unsigned long> keys(recs.size());
        for (size_t i = 0; i < recs.size(); ++i) keys[i] = recs[i].key;

        /* parallel FastFlow sort of the keys */
        t_mid = MPI_Wtime();

        keys = fastflow_sort_keys(keys, threads);

        /* rebuild the sorted records */
        std::vector<Record> final_sorted = rebuild_sorted_records(keys, recs);

        t_mid2 = MPI_Wtime();

        // ==================== [TEST CODE BEGIN] ====================
        bool ok = std::is_sorted(final_sorted.begin(), final_sorted.end(),
                                 [](const Record& a,const Record& b){ return a.key <= b.key; });
        if (!ok)
            std::cerr << "[rank " << rank << "] block is not sorted!\n";
        // ==================== [TEST CODE END] ======================

        std::string outname = "./Dataset/sorted_par_" + s_arg + "_" + r_arg + ".bin";
        write_sorted_records(outname, final_sorted, PAYLOAD_SIZE);

        t_end = MPI_Wtime();

        t_total = (t_mid2 - t_mid) + (t_end - t_mid2) + (t_mid - t_start);
        /* print times and verification */
        std::cout << "Total (get data) time: "   << t_mid - t_start << " s\n"
                    << "Total (sort) time: "     << t_mid2 - t_mid   << " s\n"
                    << "Total (write) time: "  << t_end - t_mid2  << " s\n"
                    << "TOTAL TIME: " << t_total << " s\n";

        /* cleanup and exit */
        MPI_Finalize();
        return 0;
    }

    /* =========== below only for size > 1 =========== */

    std::vector<Record> recs;
    std::vector<unsigned long> all_keys;
    int *counts = new int[size];
    int *displs = new int[size];

    t_start = MPI_Wtime();

    /* -------------------- rank 0: load ---------------- */
    if (rank == 0) {
        recs = load_records("./Dataset/data_" + s_arg + "_" + r_arg + ".bin");

        int base = N / size, rem = N % size, offset = 0;
        for (int i = 0; i < size; ++i) {
            counts[i] = base + (i < rem ? 1 : 0);
            displs[i] = offset;
            offset += counts[i];
        }

        all_keys.resize(N);
        for (size_t i = 0; i < N; ++i) all_keys[i] = recs[i].key;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------- distribute keys ------------- */
    MPI_Bcast(counts, size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_N = counts[rank];
    std::vector<unsigned long> local_keys(local_N);

    MPI_Scatterv(all_keys.data(), counts, displs, MPI_UNSIGNED_LONG,
                 local_keys.data(), local_N, MPI_UNSIGNED_LONG,
                 0, MPI_COMM_WORLD);

    t_mid = MPI_Wtime();

    /* -------------------- local sort ------------------ */
    local_keys = fastflow_sort_keys(local_keys, threads);

    /* -------------------- binary merge ---------------- */
    std::vector<unsigned long> sorted_keys = tree_merge_keys(local_keys, rank, size);

    /* -------------------- rebuild & output (rank 0) --- */
    std::vector<Record> final_sorted;
    if (rank == 0) {
        final_sorted = rebuild_sorted_records(sorted_keys, recs);

        t_mid2 = MPI_Wtime();

        // ==================== [TEST CODE BEGIN] ====================
        bool ok = std::is_sorted(final_sorted.begin(), final_sorted.end(),
                                 [](const Record& a,const Record& b){ return a.key <= b.key; });
        if (!ok)
            std::cerr << "[rank " << rank << "] block is not sorted!\n";
        // ==================== [TEST CODE END] ======================

        std::string outname = "./Dataset/sorted_par_" + s_arg + "_" + r_arg + ".bin";
        write_sorted_records(outname, final_sorted, PAYLOAD_SIZE);

        t_end = MPI_Wtime();

        t_total = (t_mid2 - t_mid) + (t_end - t_mid2) + (t_mid - t_start);
        /* print times and verification */
        std::cout << "Total (get data) time: "   << t_mid - t_start << " s\n"
                    << "Total (sort) time: "     << t_mid2 - t_mid   << " s\n"
                    << "Total (write) time: "  << t_end - t_mid2  << " s\n"
                    << "TOTAL TIME: " << t_total << " s\n";

    }

    delete[] counts;
    delete[] displs;
    MPI_Finalize();
    return 0;
}
