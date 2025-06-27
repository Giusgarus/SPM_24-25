#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <atomic>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return -1;
    }

    const char* filename = argv[1];
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return -1;
    }

    std::atomic<int> total_words(0);

#pragma omp parallel
    {
#pragma omp single nowait
        {
            std::string line;
            while (std::getline(infile, line)) {
                // Allocate memory for the line and the word count result.
                // These pointers will be used in the dependency clauses.
                std::string* line_ptr = new std::string(line);

                // Stage 1: Task that "produces" the line.
                #pragma omp task depend(out: line_ptr[0])
                {
                    // Nothing to do here
                }

                // Stage 2: Count the words in the line.
#pragma omp task depend(in: line_ptr[0]) 
                {
                    std::istringstream iss(*line_ptr);
                    int count = 0;
                    std::string word;
                    while (iss >> word) {
                        ++count;
                    }
                    total_words.fetch_add(count, std::memory_order_relaxed);
                    delete line_ptr; 
                }
            } // end while (reading file)
            // Wait for all tasks to finish
#pragma omp taskwait
        } // end single
    } // end parallel

    // Print the final total word count.
    std::cout << total_words.load() << " " << filename << "\n";
}
