#include <iostream>
#include <vector>
#include <thread>
#include <getopt.h>
#include <future>
#include <string.h>
#include <hpc_helpers.hpp>
#include <threadPool.hpp>


using ull = unsigned long long;

// Define a structure to represent a range of numbers for the Collatz computation
struct collatz_ranges {
    ull start; // The starting number of the range
    ull end;   // The ending number of the range
    size_t idx; // The index of the range, used to identify which range this structure belongs to
};



void collatz(std::vector<collatz_ranges> embedded_ranges, size_t n_ranges, std::promise<int *> && results) {
    // Allocate memory for an array to store the maximum Collatz sequence length for each range.
    int *max_collatz = new int[n_ranges];
    // Initialize the array to zero for all ranges.
    memset(max_collatz, 0, sizeof(int) * n_ranges);

    // Iterate over each range in the provided vector of ranges.
    for(const auto& range : embedded_ranges) {
        ull start = range.start; // Extract the start of the current range.
        ull end = range.end;     // Extract the end of the current range.

        // Iterate over each number in the current range [start, end].
        for (ull i = start; i <= end; ++i) {
            ull n = i; // Initialize the current number for Collatz computation.
            int steps = 0; // Initialize the step counter for the Collatz sequence.

            // Compute the Collatz sequence for the current number until it reaches 1.
            while (n != 1) {
                // If the number is even, divide it by 2; otherwise, apply 3n + 1.
                if ((n & 1) == 0) {
                    n >>= 1; // n /= 2
                } else {
                    n = (3 * n + 1) >> 1; // combina due passaggi
                    steps++; // conto il passaggio extra
                }
                steps++;
            }

            // Update the maximum Collatz sequence length for the current range index.
            max_collatz[range.idx] = std::max(max_collatz[range.idx], steps);
        }
    }

    // Set the computed results (max_collatz array) into the promise to be retrieved by the caller.
    results.set_value(max_collatz);
}





/**
 * @brief Executes the Collatz conjecture computation in parallel using a static scheduling approach.
 * 
 * This function divides the input ranges into chunks and assigns them to multiple threads for parallel
 * computation. Each thread processes its assigned chunks independently, and the results are aggregated
 * to determine the maximum Collatz sequence length for each range.
 * 
 * @param num_threads The number of threads to use for parallel computation.
 * @param chunk_size The size of each chunk to divide the ranges into for static scheduling.
 * @param ranges A vector of pairs representing the ranges [start, end] for which the Collatz computation
 *               needs to be performed.
 * @param n_ranges The total number of ranges in the `ranges` vector.
 * 
 * @details
 * - The function first divides the input ranges into smaller chunks and assigns them to threads in a 
 *   round-robin fashion based on the chunk size and the number of threads.
 * - Each thread processes its assigned chunks by invoking the `collatz` function, which computes the 
 *   maximum Collatz sequence length for the given chunks.
 * - The results from all threads are collected using `std::future` and aggregated to determine the 
 *   maximum Collatz sequence length for each range.
 * - The function ensures proper cleanup of dynamically allocated memory and joins all threads before 
 *   exiting.
 * 
 * @note
 * - The `collatz` function is expected to take a vector of `collatz_ranges`, the total number of ranges,
 *   and a `std::promise` to store the results.
 * - The `TIMERSTART` and `TIMERSTOP` macros are used to measure the time taken for specific sections of 
 *   the code (e.g., job division and computation).
 * 
 * @warning
 * - The caller is responsible for ensuring that the input parameters are valid. For example, `num_threads`
 *   should be greater than 0, `chunk_size` should be positive, and `ranges` should not be empty.
 * - The function uses dynamic memory allocation for storing intermediate results and jobs. Ensure that 
 *   sufficient memory is available to avoid allocation failures.
 */
void parallel_static_collatz(int num_threads, int chunk_size, const std::vector<std::pair<ull, ull>>& ranges, size_t n_ranges) {

    // Vector to store threads for parallel execution.
    std::vector<std::thread> threads;
    // Vector to store futures for retrieving results from threads.
    std::vector<std::future<int *>> results;

    // Start timer for job division phase.
    TIMERSTART(Static_par_time);
    // Allocate an array of vectors to store jobs for each thread.
    std::vector<collatz_ranges> *jobs = new std::vector<collatz_ranges>[num_threads];
    // Loop over each thread to assign jobs.
    for (int j = 0; j < num_threads; j++) {
        // Temporary vector to store ranges assigned to the current thread.
        std::vector<collatz_ranges> embedded_ranges;
        size_t r = 0; // Index to track the range being processed.
        // Iterate over all input ranges.
        for (const auto& range : ranges) {
            // Divide the range into chunks and assign them to the current thread.
            for (ull i = (j * chunk_size) + range.first; i <= range.second; i += num_threads * chunk_size) {
                // Calculate the end of the current chunk.
                ull chunk_end = std::min(i + chunk_size - 1, range.second);
                // Add the chunk to the current thread's job list.
                embedded_ranges.emplace_back(collatz_ranges{i, chunk_end, r});
            }
            r++; // Move to the next range.
        }
        // Assign the embedded ranges to the current thread.
        jobs[j] = embedded_ranges;
    }

    // Launch threads to process the jobs.
    for(int i = 0; i < num_threads; i++) {
        // Create a promise to store the results of the current thread.
        std::promise<int *> promise;
        // Store the future associated with the promise.
        results.emplace_back(promise.get_future());
        // Launch a thread to execute the Collatz computation for the assigned jobs.
        threads.emplace_back(
            [jobs, i, n_ranges, p{std::move(promise)}]() mutable {
                collatz(jobs[i], n_ranges, std::move(p));
            }
        );
    }

    // Allocate memory to store the final results for all ranges.
    int *final_results = new int[n_ranges];
    // Initialize the final results array to zero.
    memset(final_results, 0, sizeof(int) * n_ranges);
    
    // Collect results from all threads.
    for (auto& result : results) {
        // Retrieve the result from the future.
        int *max_collatz = result.get();
        // Update the final results with the maximum values from each thread.
        for (size_t i = 0; i < n_ranges; ++i) {
            final_results[i] = std::max(final_results[i], max_collatz[i]);
        }
        // Free the memory allocated for the thread's results.
        delete[] max_collatz;
    }

    // Stop timer for computation phase.
    TIMERSTOP(Static_par_time);

    // Print the final results for each range.
    size_t i = 0;
    for (auto& range: ranges) {
        std::cout << range.first << "-" << range.second << ": " << final_results[i] << "\n";
        i++;
    }
    // Free the memory allocated for the final results and jobs.
    delete[] final_results;
    delete[] jobs;

    // Join all threads to ensure they complete execution.
    for(auto& thread : threads) {
        thread.join();
    }
    // Clear the threads and results vectors.
    threads.clear();
    results.clear();
    // Set pointers to null for safety.
    jobs = nullptr;
    final_results = nullptr;
}


/**
 * @brief Computes the maximum number of steps required to reach 1 in the Collatz sequence
 *        for a range of numbers, using a parallel dynamic task distribution approach.
 *
 * This function divides the computation of the Collatz sequence into smaller tasks, 
 * which are distributed among multiple threads using a thread pool. Each task computes 
 * the maximum number of steps for a subrange of numbers, and the results are aggregated 
 * to determine the maximum steps for each range.
 *
 * @param num_threads The number of threads to use in the thread pool.
 * @param task_size The size of each subrange (chunk) to be processed by a single task.
 * @param ranges A vector of pairs, where each pair represents a range [start, end] of numbers 
 *               for which the Collatz sequence will be computed.
 * @param n_ranges The number of ranges in the `ranges` vector.
 *
 * The function performs the following steps:
 * 1. Initializes a thread pool with the specified number of threads.
 * 2. Defines a lambda function (`collatz_task`) to compute the maximum Collatz steps for a given range.
 * 3. Divides each range in `ranges` into smaller chunks of size `task_size` and enqueues these chunks 
 *    as tasks in the thread pool.
 * 4. Collects the results of the tasks using `std::future` and aggregates the maximum steps for each range.
 * 5. Outputs the maximum number of steps for each range in the format: "start-end: max_steps".
 *
 * @note The Collatz sequence for a number `n` is computed as follows:
 *       - If `n` is even, the next number is `n / 2`.
 *       - If `n` is odd, the next number is `3 * n + 1`.
 *       - The sequence ends when `n` becomes 1.
 *
 * @warning The function assumes that the ranges in the `ranges` vector do not overlap and are valid.
 *          Behavior is undefined if the ranges overlap or if `task_size` is zero.
 */
void parallel_dynamic_collatz(int num_threads, int task_size, std::vector<std::pair<ull, ull>>& ranges, size_t n_ranges) {

    // Create a thread pool with the specified number of threads.
    ThreadPool pool(num_threads);

    // Define a lambda function to compute the maximum Collatz steps for a given range.
    auto collatz_task = [](collatz_ranges range) {
        int max_steps = 0; // Initialize the maximum steps for the range to 0.
        // Iterate over all numbers in the range [start, end].
        for (ull i = range.start; i <= range.end; ++i) {
            ull n = i; // Start with the current number.
            int steps = 0; // Initialize the step counter for the Collatz sequence.
            // Compute the Collatz sequence until the number reaches 1.
            while (n != 1) {
                if ((n & 1) == 0) {
                    n >>= 1; // n /= 2
                } else {
                    n = (3 * n + 1) >> 1; // combina due passaggi
                    steps++; // conto il passaggio extra
                }
                steps++;
            }
            // Update the maximum steps for the range.
            max_steps = std::max(max_steps, steps);
        }
        // Return the index of the range and the maximum steps as a pair.
        return std::pair(range.idx , max_steps);
    };

    // Vector to store the futures for the results of the tasks.
    std::vector<std::future<std::pair<size_t, int>>> results;

    TIMERSTART(Dynamic_par_time)

    // Iterate over all ranges to divide them into smaller tasks.
    for (ull i = 0; i < n_ranges; ++i) {
        ull start = ranges[i].first; // Get the start of the current range.
        ull end = ranges[i].second; // Get the end of the current range.
        // Divide the range into chunks of size `task_size`.
        for (ull j = start; j <= end; j += task_size) {
            ull chunk_end = std::min(j + task_size - 1, end); // Determine the end of the current chunk.
            collatz_ranges range{j, chunk_end, i}; // Create a `collatz_ranges` object for the chunk.
            // Enqueue the task into the thread pool and store the future.
            results.emplace_back(pool.enqueue(collatz_task, range));
        }
    }

    // Vector to store the maximum Collatz steps for each range.
    std::vector<int> max_collatz(n_ranges, 0);

    // Collect the results from all tasks.
    for (auto& result : results) {
        auto [idx, max_steps] = result.get(); // Retrieve the index and maximum steps from the future.
        // Update the maximum steps for the corresponding range.
        max_collatz[idx] = std::max(max_collatz[idx], max_steps);
    }

    TIMERSTOP(Dynamic_par_time)

    // Print the maximum Collatz steps for each range.
    for (size_t i = 0; i < n_ranges; ++i) {
        std::cout << ranges[i].first << "-" << ranges[i].second << ": " << max_collatz[i] << "\n";
    }

}



int main(int argc, char* argv[]) {
    // Initialize the number of threads to 16 by default.
    int num_threads = 16;
    // Initialize the chunk size (or task size) to 32 by default.
    int size = 32;
    // Initialize the scheduling mode to static by default (dynamic = false).
    bool dynamic = false;
    // Vector to store the input ranges provided as command-line arguments.
    std::vector<std::pair<ull, ull>> ranges;

    // Variable to store the current option parsed by getopt.
    int opt;
    // Parse command-line options using getopt.
    while ((opt = getopt(argc, argv, "dn:c:")) != -1) {
        switch (opt) {
            case 'd':
                // If the '-d' flag is provided, enable dynamic scheduling.
                dynamic = true;
                break;
            case 'n':
                // If the '-n' flag is provided, set the number of threads.
                num_threads = std::stoi(optarg);
                break;
            case 'c':
                // If the '-c' flag is provided, set the chunk/task size.
                size = std::stoi(optarg);
                break;
            default:
                // If an invalid option is provided, print usage instructions and exit.
                std::cerr << "Usage: " << argv[0] << " [-d] -n <num_threads> -c <chunk_size> <range1> <range2> ..." << std::endl;
                return 1;
        }
    }

    // Parse the remaining command-line arguments as ranges.
    for (int i = optind; i < argc; ++i) {
        // Convert the current argument to a string.
        std::string range_str = argv[i];
        // Find the position of the '-' character in the range string.
        size_t dash_pos = range_str.find('-');
        if (dash_pos != std::string::npos) {
            // Extract the start of the range (substring before the '-').
            ull start = std::stoull(range_str.substr(0, dash_pos));
            // Extract the end of the range (substring after the '-').
            ull end = std::stoull(range_str.substr(dash_pos + 1));
            // Add the parsed range to the ranges vector.
            ranges.emplace_back(start, end);
        }
    }

    // Determine the total number of ranges provided.
    size_t n_ranges = ranges.size();

    // Print whether dynamic or static scheduling is being used.
    std::cout << (dynamic ? "Dynamic" : "Static") << " scheduling.\n";
    if(dynamic){
        // If dynamic scheduling is enabled, print the configuration and call the dynamic function.
        std::cout << "Running with " << num_threads << " threads and a task size of " << size << " numbers.\n";
        parallel_dynamic_collatz(num_threads, size, ranges, n_ranges);
    }
    else{
        // If static scheduling is enabled, print the configuration and call the static function.
        std::cout << "Running with " << num_threads << " threads and a block size of " << size << " numbers.\n";
        parallel_static_collatz(num_threads, size, ranges, n_ranges);
    }

    // Return 0 to indicate successful execution.
    return 0;
}