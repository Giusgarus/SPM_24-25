#include <config.hpp>
#include <cmdline.hpp>
#include <utility.hpp>
#include <hpc_helpers.hpp> 
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <numeric> 

/* -------------------------------------------------------------------------- */
/*                            data-structure aliases                           */
/* -------------------------------------------------------------------------- */

// Structure to represent a file descriptor
struct FileDescriptor
{
    std::size_t bytes;    // Size of the file on disk
    std::string filename; // Full path to the file
};

// Structure to represent a chunk of data for compression/decompression
struct Chunk
{
    std::size_t raw_bytes{};  // Size of the data before compression
    std::size_t zip_bytes{};  // Size of the data after compression
    unsigned char *payload{}; // Dynamically allocated buffer for the data
};


/**
 * @brief Checks if the given file path has a compressed file suffix.
 * 
 * This function determines whether the provided file path ends with a 
 * suffix commonly associated with compressed files (e.g., ".zip", ".gz").
 * 
 * @param path The file path to check.
 * @return true if the file path has a compressed file suffix, false otherwise.
 */
inline bool has_compressed_suffix(const std::string &path);

/**
 * @brief Enumerates the files and directories within the specified path and 
 *        stores their descriptors in the provided output vector.
 * 
 * @tparam Iter The iterator type used for traversing the file system.
 * @param p The path to the directory or file to be enumerated.
 * @param out A reference to a vector where the file descriptors will be stored.
 */
template <class Iter>
void enumerate(const std::filesystem::path &p, std::vector<FileDescriptor> &out);


/**
 * @brief Writes a compressed archive to a file.
 * 
 * This function takes the name of the original file, the total size of the 
 * compressed data, the size of the original uncompressed data, and a chunk 
 * containing the compressed data. It writes the compressed archive to the 
 * appropriate destination.
 * 
 * @param original_name The name of the original file being compressed.
 * @param total_zip_bytes The total size of the compressed data in bytes.
 * @param original_bytes The size of the original uncompressed data in bytes.
 * @param chunk A reference to a Chunk object containing the compressed data.
 */
void write_archive(const std::string &original_name,
                   std::size_t total_zip_bytes,
                   std::size_t original_bytes,
                   Chunk &chunk);


/**
 * @brief Decompresses a chunk of data using the inflate algorithm.
 *
 * This function takes a compressed data chunk and decompresses it into the
 * provided destination buffer. It is designed to handle raw deflate streams.
 *
 * @param src Pointer to the source buffer containing the compressed data.
 * @param dst_bytes The size of the destination buffer in bytes.
 * @param dst Pointer to the destination buffer where decompressed data will be stored.
 * @param src_bytes The size of the source buffer in bytes.
 * @return true if the decompression is successful, false otherwise.
 */
static bool inflate_chunk(const unsigned char *src,
                          std::size_t dst_bytes,
                          unsigned char *dst,
                          std::size_t src_bytes);


/**
 * @brief Compresses a chunk of data using the deflate algorithm.
 * 
 * @param src Pointer to the source data to be compressed.
 * @param src_bytes Size of the source data in bytes.
 * @param dst Reference to a pointer where the compressed data will be stored.
 *            The memory for the compressed data will be allocated dynamically.
 * @param dst_bytes Reference to a variable where the size of the compressed data
 *                  in bytes will be stored.
 * @return true if the compression was successful, false otherwise.
 * 
 * @note The caller is responsible for freeing the memory allocated for the
 *       compressed data pointed to by `dst`.
 */
static bool deflate_chunk(const unsigned char *src,
                          std::size_t src_bytes,
                          unsigned char *&dst,
                          std::size_t &dst_bytes);

/**
 * @brief Extracts a file from a compressed archive.
 * 
 * This function decompresses the data stored in a compressed archive and writes
 * it to a file. It handles the memory mapping and unmapping of the archive.
 * 
 * @param packed_name The name of the compressed archive file.
 * @param packed_bytes The size of the compressed archive in bytes.
 */
void extract_file(const std::string &packed_name, std::size_t packed_bytes);

/**
 * @brief Archives a file by processing its name and size.
 * 
 * This function takes the name of a file and its size in bytes, and performs
 * operations to archive the file.
 * 
 * @param plain_name The name of the file to be archived, as a string.
 * @param plain_bytes The size of the file in bytes, as a std::size_t.
 */
void archive_file(const std::string &plain_name, std::size_t plain_bytes);

/* -------------------------------------------------------------------------- */
/*                                   main                                      */
/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    // Check if at least one argument is provided
    if (argc < 2)
    {
        usage(argv[0]); // Print usage instructions
        return -1; // Exit with error code
    }

    std::vector<FileDescriptor> catalogue; // Vector to store file descriptors
    long arg_idx = parseCommandLine(argc, argv); // Parse command-line arguments
    if (arg_idx < 0) // Check for parsing errors
        return -1; // Exit with error code

    TIMERSTART(Scan_Folders); // Start timing the folder scanning process
    while (argv[arg_idx]) // Loop through remaining command-line arguments
    {
        std::filesystem::path p(argv[arg_idx++]); // Convert argument to filesystem path
        // Enumerate files in the directory (recursively or non-recursively based on RECUR)
        RECUR ? enumerate<std::filesystem::recursive_directory_iterator>(p, catalogue)
              : enumerate<std::filesystem::directory_iterator>(p, catalogue);
    }
    TIMERSTOP(Scan_Folders); // Stop timing the folder scanning process

    TIMERSTART(Processing); // Start timing the processing phase

    // Loop through the catalogue of files
    for (std::size_t i = 0; i < catalogue.size(); ++i)
    {
        // Determine whether to skip the file based on its compression status
        bool skip = COMP ? has_compressed_suffix(catalogue[i].filename)   // If compressing, skip already compressed files
                         : !has_compressed_suffix(catalogue[i].filename); // If decompressing, skip non-compressed files
        if (skip)
            continue; // Skip the current file

        // Compress or decompress the file based on the COMP flag
        COMP ? archive_file(catalogue[i].filename, catalogue[i].bytes)
             : extract_file(catalogue[i].filename, catalogue[i].bytes);
    }

    TIMERSTOP(Processing); // Stop timing the processing phase
    return 0; // Exit successfully
}

/* -------------------------------------------------------------------------- */
/*                        utility: riconoscere file .zip                      */
/* -------------------------------------------------------------------------- */

// Function to check if a file has a specific compressed suffix
inline bool has_compressed_suffix(const std::string &path)
{
    constexpr std::size_t suff_len = std::strlen(SUFFIX); // Length of the suffix (e.g., ".zip")
    return path.size() >= suff_len && // Ensure the path is long enough to contain the suffix
           path.compare(path.size() - suff_len, suff_len, SUFFIX) == 0; // Compare the suffix with the end of the path
}

/* -------------------------------------------------------------------------- */
/*                             directory traversal                             */
/* -------------------------------------------------------------------------- */

// Template function to enumerate files in a directory
template <class Iter>
void enumerate(const std::filesystem::path &p, std::vector<FileDescriptor> &out)
{
    // If the path is a regular file, add it to the output vector
    if (std::filesystem::is_regular_file(p))
    {
        out.push_back({std::filesystem::file_size(p), p.string()});
        return;
    }
    // Iterate through the directory entries
    for (const auto &entry : Iter(p))
    {
        // If the entry is a regular file, add it to the output vector
        if (entry.is_regular_file())
        {
            out.push_back({entry.file_size(), entry.path().string()});
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                         low-level (de)compression I/O                       */
/* -------------------------------------------------------------------------- */

// Function to write a compressed archive to disk
void write_archive(const std::string &original_name,
                   std::size_t total_zip_bytes,
                   std::size_t original_bytes,
                   Chunk &chunk)
{
    const std::size_t header_bytes =
        sizeof(std::size_t) * (2 + 2); // Size of the header (size + nChunks + (raw,zip)*n)

    const std::size_t final_bytes = header_bytes + total_zip_bytes; // Total size of the archive
    unsigned char *map_out = nullptr; // Pointer for memory-mapped output

    // Allocate space for the archive file
    if (!allocateFile(std::string(original_name + SUFFIX).c_str(), final_bytes, map_out))
    {
        std::cerr << "cannot create archive" << std::endl; // Print error if allocation fails
        return;
    }

    /* ---------- header --------- */
    unsigned char *cursor = map_out; // Pointer to the current position in the output buffer
    auto emit = [&](auto val)
    {
        std::memcpy(cursor, &val, sizeof(val)); // Copy value to the output buffer
        cursor += sizeof(val); // Advance the cursor
    };

    emit(original_bytes); // Write the original size of the file

    emit(chunk.raw_bytes); // Write the raw size of the chunk
    emit(chunk.zip_bytes); // Write the compressed size of the chunk

    /* ---------- body ----------- */

    std::memcpy(cursor, chunk.payload, chunk.zip_bytes); // Copy the compressed data to the output buffer
    delete[] chunk.payload; // Free the dynamically allocated buffer

    unmapFile(map_out, final_bytes); // Unmap the output file
    if (REMOVE_ORIGIN) // If the REMOVE_ORIGIN flag is set
        unlink(original_name.c_str()); // Delete the original file
}

// Function to decompress a chunk of data
static bool inflate_chunk(const unsigned char *src,
                          std::size_t dst_bytes,
                          unsigned char *dst,
                          std::size_t src_bytes)
{
    uLongf len = dst_bytes; // Length of the decompressed data
    // Attempt to decompress the data
    if (uncompress(dst, &len, src, static_cast<uLong>(src_bytes)) != Z_OK)
    {
        if (QUITE_MODE >= 1) // If quiet mode is not enabled
            std::cerr << "inflate failed\n"; // Print an error message
        return false; // Return failure
    }
    return true; // Return success
}

// Function to compress a chunk of data
static bool deflate_chunk(const unsigned char *src,
                          std::size_t src_bytes,
                          unsigned char *&dst,
                          std::size_t &dst_bytes)
{
    uLongf max = compressBound(static_cast<uLong>(src_bytes)); // Calculate the maximum compressed size
    dst = new unsigned char[max]; // Allocate memory for the compressed data
    // Attempt to compress the data
    if (compress(dst, &max, src, static_cast<uLong>(src_bytes)) != Z_OK)
    {
        if (QUITE_MODE >= 1) // If quiet mode is not enabled
            std::cerr << "deflate failed\n"; // Print an error message
        delete[] dst; // Free the allocated memory
        return false; // Return failure
    }
    dst_bytes = static_cast<std::size_t>(max); // Update the compressed size
    return true; // Return success
}

/* -------------------------------------------------------------------------- */
/*                               decompression                                 */
/* -------------------------------------------------------------------------- */

// Function to extract a file from a compressed archive
void extract_file(const std::string &packed_name, std::size_t packed_bytes)
{
    /* Map the archive file into memory */
    unsigned char *map_in = nullptr;
    if (!mapFile(packed_name.c_str(), packed_bytes, map_in))
    {
        std::cerr << "cannot map " << packed_name << std::endl; // Print error if mapping fails
        return;
    }

    /* ----- Read header ----- */
    const unsigned char *rdr = map_in; // Pointer to the current position in the input buffer
    std::size_t fresh_bytes{}; // Size of the decompressed file
    auto pull = [&](auto &v)
    {
        std::memcpy(&v, rdr, sizeof(v)); // Copy value from the input buffer
        rdr += sizeof(v); // Advance the pointer
    };
    pull(fresh_bytes); // Read the size of the decompressed file

    Chunk ck{}; // Initialize a chunk structure
    pull(ck.raw_bytes); // Read the raw size of the chunk
    pull(ck.zip_bytes); // Read the compressed size of the chunk

    /* Build offset tables */
    std::size_t zip_offsets = rdr - map_in; // Calculate the offset to the compressed data

    /* Prepare output file */
    const std::string fresh_name = packed_name.substr(0, packed_name.size() - strlen(SUFFIX)); // Remove the suffix from the filename
    unsigned char *map_out = nullptr; // Pointer for memory-mapped output
    if (!allocateFile(fresh_name.c_str(), fresh_bytes, map_out))
    {
        std::cerr << "cannot alloc for output" << std::endl; // Print error if allocation fails
        return;
    }

    // Decompress the data
    inflate_chunk(map_in + zip_offsets,
                  ck.raw_bytes,
                  map_out,
                  ck.zip_bytes);

    unmapFile(map_in, packed_bytes); // Unmap the input file
    unmapFile(map_out, fresh_bytes); // Unmap the output file
    if (REMOVE_ORIGIN) // If the REMOVE_ORIGIN flag is set
        unlink(packed_name.c_str()); // Delete the original archive
}

/* -------------------------------------------------------------------------- */
/*                                 compression                                 */
/* -------------------------------------------------------------------------- */

// Function to compress a file into an archive
void archive_file(const std::string &plain_name, std::size_t plain_bytes)
{
    unsigned char *map_in = nullptr; // Pointer for memory-mapped input
    if (!mapFile(plain_name.c_str(), plain_bytes, map_in))
    {
        std::cerr << "cannot map " << plain_name << std::endl; // Print error if mapping fails
        return;
    }

    std::size_t total_zip_bytes = 0; // Total size of the compressed data

    Chunk ck{}; // Initialize a chunk structure
    // Attempt to compress the data
    if (deflate_chunk(map_in, plain_bytes, ck.payload, ck.zip_bytes))
    {
        ck.raw_bytes = plain_bytes; // Set the raw size of the chunk
        total_zip_bytes = ck.zip_bytes; // Update the total compressed size
    }

    unmapFile(map_in, plain_bytes); // Unmap the input file
    write_archive(plain_name, total_zip_bytes, plain_bytes, ck); // Write the compressed archive to disk
    if (REMOVE_ORIGIN) // If the REMOVE_ORIGIN flag is set
        unlink(plain_name.c_str()); // Delete the original file
}
