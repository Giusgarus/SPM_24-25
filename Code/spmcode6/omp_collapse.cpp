#include <iostream>
#include <omp.h>

int main() {

	auto print=[](int i, int j) {
		std::printf("Thread%d  (%d,%d)\n",
					omp_get_thread_num(), i, j);
		
	};
	
#pragma omp parallel for schedule(static) num_threads(5) collapse(2) 
	for (int i = 0; i < 4; ++i)  
		for(int j=0; j < 5; ++j) 
			print(i,j); 

}
