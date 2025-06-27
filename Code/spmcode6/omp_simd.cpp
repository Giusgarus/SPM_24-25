#include <iostream>
#include <vector>
#include <omp.h>

int main() {

    const int N = 100000000;
    std::vector<double> A(N), B(N);
    
    for (int i = 0; i < N; ++i) {
        A[i] = 1.0;  
        B[i] = 2.0;  
    }


    double start = omp_get_wtime();      
    double sum = 0.0;

#pragma omp parallel for simd reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += A[i] * B[i];
    }

    double elapsed = omp_get_wtime() - start;    
    std::cout << "Result: " << sum << " time: " << elapsed << "s\n";
}


