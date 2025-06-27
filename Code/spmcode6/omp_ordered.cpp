#include <iostream>
#include <chrono>
#include <thread>
#include <omp.h>


int main() {
    const int n = 20;

    // ordered loop, we can use an ordered region inside it
#pragma omp parallel for ordered schedule(static) num_threads(4)
    for (int i = 0; i < n; i++) {

        int square = i * i;

		int tid= omp_get_thread_num();
		if (tid == 1 || tid == 3) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
        
        // It ensures that this block of code is executed in loop order
        #pragma omp ordered
        {
			std::printf("Iteration %d: %d\n", i, square);
        }
    }
    
    return 0;
}
