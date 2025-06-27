#include <cstdio>
#include <string>


long fib( long n ) {
    long n1, n2;
    if (n < 2) return 1;
	
#pragma omp task shared(n1) 
	n1 = fib(n-1);
#pragma omp task shared(n2) 
	n2 = fib(n-2);
	// Wait for the two tasks to complete
#pragma omp taskwait
	return n1 + n2;
}

int main(int argc, char *argv[]) {
    if (argc!=3){
		std::printf("Use %s n #threads\n",argv[0]);
		return -1;
	}
	long n   = std::stol(argv[1]);
	long nth = std::stol(argv[2]); 
	
	long result;
    // creating a pool of nth threads
#pragma omp parallel num_threads(nth)
    {
		// only the Master thread starts the recursion
#pragma omp master
        result = fib(n);
    }
	std::printf("fib(%ld) = %ld\n", n, result);
}

