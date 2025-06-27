#include <iostream>
#include <limits>
#include <iomanip>
#include <functional>
#include <omp.h>

using namespace std;

int main(int argc, char * argv[]) {
  if(argc != 2) {
     std::cout << "Usage is: " << argv[0] << " num_steps\n";
     return(-1);
  }

  uint64_t num_steps = std::stol(argv[1]);
  long double step= 1.0/num_steps;
  constexpr uint64_t THREASHOLD=1024;

  // this is because we should declara pi_comp before using it
  // within the body of the lambda for the recursion
  std::function<long double(uint64_t, uint64_t)> pi_comp;
  
  pi_comp = [step, &pi_comp](uint64_t start, uint64_t stop) -> long double {

	  std::printf("(%ld, %ld) diff = %ld\n", start, stop, (stop-start));
	  
	  if (stop-start < THREASHOLD) {
		  long double sum = 0.0;
		  long double   x = 0.0;
		  std::printf("compute (%ld, %ld)\n", start, stop);
		  for (uint64_t k=start; k < stop; ++k) {
			  x = (k + 0.5) * step;
			  sum += 4.0/(1.0 + x*x);
		  }
		  return sum;
	  } 
	  int tmp = stop-start;
	  auto sum1 = pi_comp(start, stop - tmp/2);
	  auto sum2 = pi_comp(stop-tmp/2, stop);
	  return (sum1 + sum2);
  };

  
  double start = omp_get_wtime();
  long double sum = 0.0;
  {
	  sum = pi_comp(0, num_steps);
  }
  long double pi = step * sum;

  double elapsed = omp_get_wtime() - start;
  
  std::cout << "Pi = " << std::setprecision(std::numeric_limits<long double>::digits10 +1) << pi << "\n";
  std::cout << "Pi = 3.141592653589793238 (first 18 decimal digits)\n";
  std::printf("Time %f (ms)\n", elapsed*1000.0);
  return(0);
}

