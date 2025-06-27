#include <iostream>
#include <vector>
#include <getopt.h>
#include <hpc_helpers.hpp>



using ull = unsigned long long;

int collatz_steps(ull start, ull end){
    int max_steps = 0;
    for (ull i = start; i <= end; ++i) {
        ull n = i;
        int steps = 0;
        while (n != 1) {
            if ((n & 1) == 0) {
                n >>= 1; // n /= 2
            } else {
                n = (3 * n + 1) >> 1; // combina due passaggi
                steps++; // conto il passaggio extra
            }
            steps++;
        }
        max_steps = std::max(max_steps, steps);
    }
    return max_steps;
}


int main(int argc, char* argv[]) {
    std::vector<std::pair<ull, ull>> ranges;

    // Check that at least one range argument is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <range1> [<range2> ...]" << std::endl;
        return 1;
    }

    // Iterate over command-line arguments, skipping the first (program name)
    for (int i = 1; i < argc; ++i) {
        std::string range_str = argv[i];
        size_t dash_pos = range_str.find('-');

        if (dash_pos != std::string::npos) {
            try {
                ull start = std::stoull(range_str.substr(0, dash_pos));
                ull end = std::stoull(range_str.substr(dash_pos + 1));

                if (start > end) {
                    std::cerr << "Error: invalid range (" << start << " > " << end << ")" << std::endl;
                    return 1;
                }

                ranges.emplace_back(start, end);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing range: " << range_str << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: invalid range format (" << range_str << ")" << std::endl;
            return 1;
        }
    }

    TIMERSTART(TOTAL);
    // Check that at least one range was added
    if (ranges.empty()) {
        std::cerr << "Error: no valid ranges provided" << std::endl;
        return 1;
    }

    // Call the collatz_steps function for each range
    std::vector<int> max_steps;
    for (const auto& range : ranges) {
        max_steps.push_back(collatz_steps(range.first, range.second));
    }

    TIMERSTOP(TOTAL);

    // Print the results
    for (size_t i = 0; i < ranges.size(); ++i) {
        std::cout << ranges[i].first << "-" << ranges[i].second << ": " << max_steps[i] << "\n";
    }

}
