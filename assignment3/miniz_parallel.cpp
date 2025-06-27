#include <config.hpp>
#include <cmdline.hpp>
#include <utility.hpp>
#include <hpc_helpers.hpp>
#include <omp.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <numeric>

// Constants for memory size units
constexpr std::size_t KB = 1 << 10; // 1 KB = 1024 bytes
constexpr std::size_t MB = 1 << 20; // 1 MB = 1024 * 1024 bytes

// Thresholds for file processing
constexpr std::size_t HUGE_FILE_THRESHOLD = 128 * MB;  // Files larger than 128MB are considered "huge"
constexpr std::size_t DEFAULT_CHUNK_CAPACITY = 256 * KB; // Default chunk size for processing is 256KB

/* -------------------------------------------------------------------------- */
/*                            Data-structure aliases                          */
/* -------------------------------------------------------------------------- */

// Represents a file with its size and full path
struct FileDescriptor
{
    std::size_t bytes;    // File size in bytes
    std::string filename; // Full file path
};

// Represents a chunk of data for compression/decompression
struct Chunk
{
    std::size_t raw_bytes{};  // Size of the chunk before compression
    std::size_t zip_bytes{};  // Size of the chunk after compression
    unsigned char *payload{}; // Dynamically allocated buffer for the chunk
};


/**
 * @brief Checks if the given file path has a compressed file suffix.
 * 
 * This function determines whether the provided file path ends with a 
 * suffix typically associated with compressed files (e.g., ".zip", ".gz").
 * 
 * @param path The file path to check.
 * @return true If the file path has a compressed file suffix.
 * @return false Otherwise.
 */
inline bool has_compressed_suffix(const std::string &path);


/**
 * @brief Enumerates the files and directories within the specified path and 
 *        stores their descriptors in the provided output vector.
 * 
 * @tparam Iter The iterator type used for traversing the file system.
 * @param p The path to the directory or file to be enumerated.
 * @param out A reference to a vector where the file descriptors will be stored.
 *            Each descriptor contains metadata about a file or directory.
 * 
 * @note This function requires the <filesystem> library for path handling and 
 *       assumes that the FileDescriptor type is defined elsewhere in the codebase.
 */
template <class Iter>
void enumerate(const std::filesystem::path &p, std::vector<FileDescriptor> &out);


/**
 * @brief Writes a compressed archive to a file.
 *
 * This function takes the compressed data stored in chunks and writes it
 * to a file with the specified name. It also provides information about
 * the total size of the compressed data and the original uncompressed size.
 *
 * @param original_name The name of the output archive file.
 * @param total_zip_bytes The total size of the compressed data in bytes.
 * @param original_bytes The total size of the original uncompressed data in bytes.
 * @param pack A vector of chunks containing the compressed data to be written.
 */
void write_archive(const std::string &original_name,
                   std::size_t total_zip_bytes,
                   std::size_t original_bytes,
                   std::vector<Chunk> &pack);

// Decompresses a chunk of data
/**
 * @brief Decompresses a chunk of data using the inflate algorithm.
 * 
 * This function takes a compressed data chunk and decompresses it into the 
 * provided destination buffer. It is designed to handle a specific portion 
 * of data and assumes that the source and destination buffers are properly 
 * allocated and sized.
 * 
 * @param src Pointer to the source buffer containing the compressed data.
 * @param dst_bytes The size of the destination buffer in bytes.
 * @param dst Pointer to the destination buffer where decompressed data will be stored.
 * @param src_bytes The size of the source buffer in bytes.
 * @return true if the decompression was successful, false otherwise.
 */
static bool inflate_chunk(const unsigned char *src,
                          std::size_t dst_bytes,
                          unsigned char *dst,
                          std::size_t src_bytes);


/**
 * @brief Compresses a chunk of data using the deflate algorithm.
 *
 * This function takes a source buffer containing uncompressed data and compresses it
 * into a destination buffer using the deflate algorithm. The destination buffer is
 * dynamically allocated and its size is updated to reflect the size of the compressed data.
 *
 * @param src Pointer to the source buffer containing the data to be compressed.
 * @param src_bytes The size, in bytes, of the source buffer.
 * @param dst Reference to a pointer that will point to the dynamically allocated
 *            destination buffer containing the compressed data.
 * @param dst_bytes Reference to a variable that will hold the size, in bytes, of the
 *                  compressed data in the destination buffer.
 * @return true if the compression was successful, false otherwise.
 *
 * @note The caller is responsible for freeing the memory allocated for the destination buffer.
 */
static bool deflate_chunk(const unsigned char *src,
                          std::size_t src_bytes,
                          unsigned char *&dst,
                          std::size_t &dst_bytes);


/**
 * @brief Extracts a file from a packed archive.
 * 
 * This function is responsible for extracting a file from a packed archive
 * given its name and size in bytes. It performs the necessary operations
 * to locate and unpack the specified file.
 * 
 * @param packed_name The name of the packed file to be extracted.
 * @param packed_bytes The size of the packed file in bytes.
 */
void extract_file(const std::string &packed_name, std::size_t packed_bytes);


/**
 * @brief Archives a file by processing its name and size.
 * 
 * This function takes the name of a file and its size in bytes, 
 * and performs operations to archive the file.
 * 
 * @param plain_name The name of the file to be archived, as a string.
 * @param plain_bytes The size of the file in bytes, as a std::size_t.
 */
void archive_file(const std::string &plain_name, std::size_t plain_bytes);

/* -------------------------------------------------------------------------- */
/*                                   Main                                      */
/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    // Check if the program has enough arguments
    if (argc < 2)
    {
        usage(argv[0]); // Print usage instructions
        return -1;
    }

    std::vector<FileDescriptor> catalogue; // List of files to process
    long arg_idx = parseCommandLine(argc, argv); // Parse command-line arguments
    if (arg_idx < 0)
        return -1;

    // Print OpenMP thread usage and default chunk size
    std::cout << "[OpenMP] USING UP TO " << omp_get_max_threads() << " THREADS\n";
    std::cout << "Default chunk size: " << DEFAULT_CHUNK_CAPACITY / KB << "KB\n";

    // Scan folders and populate the file catalogue
    TIMERSTART(Scan_Folders);
    while (argv[arg_idx])
    {
        std::filesystem::path p(argv[arg_idx++]);
        RECUR ? enumerate<std::filesystem::recursive_directory_iterator>(p, catalogue)
              : enumerate<std::filesystem::directory_iterator>(p, catalogue);
    }
    TIMERSTOP(Scan_Folders);

    // Process files in parallel using OpenMP
    TIMERSTART(Processing);
#pragma omp parallel
    {
#pragma omp single
        {
            for (std::size_t i = 0; i < catalogue.size(); ++i)
            {
                // Determine whether to skip the file based on its compression status
                bool skip = COMP ? has_compressed_suffix(catalogue[i].filename)   // Skip already compressed files
                                 : !has_compressed_suffix(catalogue[i].filename); // Skip non-compressed files
                if (skip)
                    continue;

                // Create a task for compressing or decompressing the file
#pragma omp task firstprivate(i)
                {
                    COMP ? archive_file(catalogue[i].filename, catalogue[i].bytes)
                         : extract_file(catalogue[i].filename, catalogue[i].bytes);
                }
            }
#pragma omp taskwait // Wait for all tasks to complete
        }
    }
    TIMERSTOP(Processing);
    return 0;
}

/* -------------------------------------------------------------------------- */
/*                        Utility: Recognize .zip files                       */
/* -------------------------------------------------------------------------- */

// Checks if a file has the specified compressed suffix
inline bool has_compressed_suffix(const std::string &path)
{
    constexpr std::size_t suff_len = std::strlen(SUFFIX); // Length of the suffix
    return path.size() >= suff_len &&
           path.compare(path.size() - suff_len, suff_len, SUFFIX) == 0;
}

/* -------------------------------------------------------------------------- */
/*                             Directory Traversal                            */
/* -------------------------------------------------------------------------- */

// Enumerates files in a directory and adds them to the output vector
template <class Iter>
void enumerate(const std::filesystem::path &p, std::vector<FileDescriptor> &out)
{
    if (std::filesystem::is_regular_file(p))
    {
        out.push_back({std::filesystem::file_size(p), p.string()});
        return;
    }
    for (const auto &entry : Iter(p))
    {
        if (entry.is_regular_file())
        {
            out.push_back({entry.file_size(), entry.path().string()});
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                         Low-level (De)compression I/O                      */
/* -------------------------------------------------------------------------- */

// Writes a compressed archive to disk
void write_archive(const std::string &original_name,
                   std::size_t total_zip_bytes,
                   std::size_t original_bytes,
                   std::vector<Chunk> &pack)
{
    // Calculate the size of the archive header
    const std::size_t header_bytes =
        sizeof(std::size_t) * (2 + pack.size() * 2); // size + nChunks + (raw,zip)*n

    const std::size_t final_bytes = header_bytes + total_zip_bytes;
    unsigned char *map_out = nullptr;

    // Allocate space for the archive
    if (!allocateFile(std::string(original_name + SUFFIX).c_str(), final_bytes, map_out))
    {
        std::cerr << "cannot create archive" << std::endl;
        return;
    }

    // Write the header
    unsigned char *cursor = map_out;
    auto emit = [&](auto val)
    {
        std::memcpy(cursor, &val, sizeof(val));
        cursor += sizeof(val);
    };

    emit(original_bytes);
    emit(static_cast<std::size_t>(pack.size()));
    for (const Chunk &c : pack)
    {
        emit(c.raw_bytes);
        emit(c.zip_bytes);
    }

    // Write the compressed data
    if (pack.size() == 1)
    {
        std::memcpy(cursor, pack[0].payload, pack[0].zip_bytes);
        delete[] pack[0].payload;
    }
    else
    {
#pragma omp taskgroup
        {
            for (std::size_t i = 0; i < pack.size(); ++i)
            {
#pragma omp task firstprivate(i)
                {
                    std::memcpy(cursor + std::accumulate(pack.begin(),
                                                         pack.begin() + i,
                                                         std::size_t(0),
                                                         [](std::size_t s, const Chunk &c)
                                                         { return s + c.zip_bytes; }),
                                pack[i].payload,
                                pack[i].zip_bytes);
                    delete[] pack[i].payload;
                }
            }
        }
    }

    unmapFile(map_out, final_bytes);
    if (REMOVE_ORIGIN)
        unlink(original_name.c_str());
}

// Decompresses a chunk of data
static bool inflate_chunk(const unsigned char *src,
                          std::size_t dst_bytes,
                          unsigned char *dst,
                          std::size_t src_bytes)
{
    uLongf len = dst_bytes;
    if (uncompress(dst, &len, src, static_cast<uLong>(src_bytes)) != Z_OK)
    {
        if (QUITE_MODE >= 1)
            std::cerr << "inflate failed\n";
        return false;
    }
    return true;
}

// Compresses a chunk of data
static bool deflate_chunk(const unsigned char *src,
                          std::size_t src_bytes,
                          unsigned char *&dst,
                          std::size_t &dst_bytes)
{
    uLongf max = compressBound(static_cast<uLong>(src_bytes));
    dst = new unsigned char[max];
    if (compress(dst, &max, src, static_cast<uLong>(src_bytes)) != Z_OK)
    {
        if (QUITE_MODE >= 1)
            std::cerr << "deflate failed\n";
        delete[] dst;
        return false;
    }
    dst_bytes = static_cast<std::size_t>(max);
    return true;
}

/* -------------------------------------------------------------------------- */
/*                               Decompression                                */
/* -------------------------------------------------------------------------- */

// Extracts a file from a compressed archive
void extract_file(const std::string &packed_name, std::size_t packed_bytes)
{
    // Map the archive into memory
    unsigned char *map_in = nullptr;
    if (!mapFile(packed_name.c_str(), packed_bytes, map_in))
    {
        std::cerr << "cannot map " << packed_name << std::endl;
        return;
    }

    // Read the archive header
    const unsigned char *rdr = map_in;
    std::size_t fresh_bytes{}, chunk_count{};
    auto pull = [&](auto &v)
    {
        std::memcpy(&v, rdr, sizeof(v));
        rdr += sizeof(v);
    };
    pull(fresh_bytes);
    pull(chunk_count);

    std::vector<Chunk> bundle(chunk_count);
    for (Chunk &c : bundle)
    {
        pull(c.raw_bytes);
        pull(c.zip_bytes);
    }

    // Build offset tables for compressed and raw data
    std::vector<std::size_t> zip_offsets(chunk_count), raw_offsets(chunk_count);
    std::size_t acc_zip = 0, acc_raw = 0;
    for (std::size_t i = 0; i < chunk_count; ++i)
    {
        zip_offsets[i] = rdr - map_in + acc_zip;
        raw_offsets[i] = acc_raw;
        acc_zip += bundle[i].zip_bytes;
        acc_raw += bundle[i].raw_bytes;
    }

    // Prepare the output file
    const std::string fresh_name = packed_name.substr(0, packed_name.size() - strlen(SUFFIX));
    unsigned char *map_out = nullptr;
    if (!allocateFile(fresh_name.c_str(), fresh_bytes, map_out))
    {
        std::cerr << "cannot alloc for output" << std::endl;
        return;
    }

    // Decompress the data
    if (chunk_count == 1)
    {
        inflate_chunk(map_in + zip_offsets[0],
                      bundle[0].raw_bytes,
                      map_out,
                      bundle[0].zip_bytes);
    }
    else
    {
        #pragma omp taskgroup
        {
            for (std::size_t i = 0; i < chunk_count; ++i)
            {
                #pragma omp task firstprivate(i)
                {
                    inflate_chunk(map_in + zip_offsets[i],
                                  bundle[i].raw_bytes,
                                  map_out + raw_offsets[i],
                                  bundle[i].zip_bytes);
                }
            }
        }
    }

    unmapFile(map_in, packed_bytes);
    unmapFile(map_out, fresh_bytes);
    if (REMOVE_ORIGIN)
        unlink(packed_name.c_str());
}

/* -------------------------------------------------------------------------- */
/*                                 Compression                                */
/* -------------------------------------------------------------------------- */

// Compresses a file into an archive
void archive_file(const std::string &plain_name, std::size_t plain_bytes)
{
    // Map the input file into memory
    unsigned char *map_in = nullptr;
    if (!mapFile(plain_name.c_str(), plain_bytes, map_in))
    {
        std::cerr << "cannot map " << plain_name << std::endl;
        return;
    }

    // Determine chunking strategy based on file size
    const bool single_chunk = (plain_bytes < HUGE_FILE_THRESHOLD);
    const std::size_t chunk_capacity = single_chunk ? plain_bytes : DEFAULT_CHUNK_CAPACITY;
    const std::size_t chunk_count =
        single_chunk ? 1 : (plain_bytes + chunk_capacity - 1) / chunk_capacity;

    std::vector<Chunk> bundle(chunk_count);
    std::size_t total_zip_bytes = 0;

    // Compress the file
    if (chunk_count == 1)   // Single chunk: no parallelism
    {
        Chunk& ck = bundle[0];
        if (deflate_chunk(map_in, plain_bytes, ck.payload, ck.zip_bytes))
        {
            ck.raw_bytes = plain_bytes;
            total_zip_bytes = ck.zip_bytes;
        }
    }
    else                  // Multiple chunks: parallelism with OpenMP tasks
    {
        #pragma omp taskgroup task_reduction(+:total_zip_bytes)
        {
            for (std::size_t i = 0; i < chunk_count; ++i)
            {
                #pragma omp task firstprivate(i) in_reduction(+:total_zip_bytes) shared(map_in, plain_bytes, chunk_capacity, bundle)
                {
                    std::size_t offset = i * chunk_capacity;
                    std::size_t current = std::min(chunk_capacity, plain_bytes - offset);
                    Chunk &ck = bundle[i];

                    if (deflate_chunk(map_in + offset, current,
                                    ck.payload, ck.zip_bytes))
                    {
                        ck.raw_bytes = current;

                    total_zip_bytes += ck.zip_bytes;
                    }
                }
            }
        }
    }

    unmapFile(map_in, plain_bytes);
    write_archive(plain_name, total_zip_bytes, plain_bytes, bundle);
    if (REMOVE_ORIGIN)
        unlink(plain_name.c_str());
}
