#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <cstdlib>  
#include <omp.h>


// Function to print the CPU affinity mask for the calling thread
// using non-portable POSIX calls. 
void print_affinity_mask() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    int ret = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if(ret != 0) {
        std::cerr << "Error retrieving CPU affinity mask\n";
        return;
    }
	std::printf("Thread %d affinity mask: ", omp_get_thread_num());

    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &cpuset)) {
			std::printf("%d ", cpu);
        }
    }
	std::printf("\n");
}

int main() {
	// Display the current thread affinity settings if defined
	const char* omp_places = std::getenv("OMP_PLACES");
	const char* omp_proc_bind = std::getenv("OMP_PROC_BIND");
	
	if (omp_places)
		std::cout << "OMP_PLACES: " << omp_places << std::endl;
	else
		std::cout << "OMP_PLACES not set" << std::endl;
	
	if (omp_proc_bind)
		std::cout << "OMP_PROC_BIND: " << omp_proc_bind << std::endl;
	else
		std::cout << "OMP_PROC_BIND not set" << std::endl;
	
#pragma omp parallel
    {
#pragma omp critical
		print_affinity_mask();
    }
    return 0;
}
