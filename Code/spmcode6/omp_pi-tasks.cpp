#include <iostream>
#include <limits>
#include <iomanip>
#include <functional>
#include <omp.h>

using namespace std;

constexpr uint64_t THREASHOLD=1024*256;

long double pi_comp(uint64_t start, uint64_t stop,
					const long double step) {
	
	if (stop-start < THREASHOLD) {
		long double sum = 0.0;
		long double   x = 0.0;
		for (uint64_t k=start; k < stop; ++k) {
			x = (k + 0.5) * step;
			sum += 4.0/(1.0 + x*x);
		}
		return sum;
	} 
	int tmp = stop-start;
	long double sum1, sum2;
#pragma omp task shared(sum1)
	sum1 = pi_comp(start, stop - tmp/2, step);
#pragma omp task shared(sum2)
	sum2 = pi_comp(stop-tmp/2, stop, step);
#pragma omp taskwait
	return (sum1 + sum2);
}


int main(int argc, char * argv[]) {
  if(argc != 2) {
     std::cout << "Usage is: " << argv[0] << " num_steps\n";
     return(-1);
  }

  uint64_t num_steps = std::stol(argv[1]);
  long double step= 1.0/num_steps;

  
  double start = omp_get_wtime();
  long double sum = 0.0;
#pragma omp parallel
  {
	  #pragma omp single
	  sum = pi_comp(0, num_steps, step);
  }
  long double pi = step * sum;

  double elapsed = omp_get_wtime() - start;
  
  std::cout << "Pi = " << std::setprecision(std::numeric_limits<long double>::digits10 +1) << pi << "\n";
  std::cout << "Pi = 3.141592653589793238 (first 18 decimal digits)\n";
  std::printf("Time %f (ms)\n", elapsed*1000.0);
  return(0);
}

