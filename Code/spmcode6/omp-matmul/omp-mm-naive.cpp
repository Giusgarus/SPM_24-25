#include <iostream>
#include <random>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "hpc_helpers.hpp"


void init(float * data, uint64_t length) {
    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1, 1);

    for (uint64_t i = 0; i < length; i++)
        data[i] = density(engine);
}

int main() {
	// assume square matrices for simplicity
	constexpr int N=4096;
    static float A[N * N];
    static float B[N * N];
    static float C[N * N];

	init(A, N*N);
	init(B, N*N);

    TIMERSTART(mm_naive1_openmp);
    // Offload the matrix multiplication to the GPU
#pragma omp target teams distribute parallel for collapse(2)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double sum = 0.0;
			for (int k = 0; k < N; k++) {
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}

    TIMERSTOP(mm_naive1_openmp);

#if 0
    float *check = new float[N*N];
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            check[i*N + j] = 0;
            for (int k=0; k<N; k++)
                check[i*N + j] += A[i*N + k] * B[k*N + j];
            if (std::abs(C[i*N+j] - check[i*N+j]) > 1e-3f) {
                std::cout << "Result error: " << C[i*N + j] << " expected" << check[i*N + j] << std::endl;
                abort();
            }
        }
    }
    std::cout << "Result is ok!" << std::endl;
    delete [] check;
#else
	(void)C;
#endif
}
