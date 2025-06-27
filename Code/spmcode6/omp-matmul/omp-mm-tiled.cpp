#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm> 
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
	// Define block (tile) size: 16 or 32 are common choices
	constexpr int BLOCK_SIZE=32;

    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

	init(A, N*N);
	init(B, N*N);

    TIMERSTART(mm_tiled_openmp);

    // Offload the matrices to the GPU. The 'target data' directive
	// maps A and B to the device, while C will be mapped back
	// from the device after computation
    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {
        // Outer loops are over tile indices, the iteration space
		// is distributed among the GPU teams
        #pragma omp target teams distribute parallel for collapse(2)
        for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                // For each tile block C[ii:ii+BLOCK_SIZE][jj:jj+BLOCK_SIZE]
                // we iterate over tiles along the k-dimension
                for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                    // Determine tile boundaries
                    int i_max = std::min(ii + BLOCK_SIZE, N);
                    int j_max = std::min(jj + BLOCK_SIZE, N);
                    int k_max = std::min(kk + BLOCK_SIZE, N);

                    for (int i = ii; i < i_max; i++) {
                        for (int j = jj; j < j_max; j++) {
                            float sum = 0.0;
                            for (int k = kk; k < k_max; k++) {
                                sum += A[i * N + k] * B[k * N + j];
                            }
                            // Accumulate partial results
                            C[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }

    TIMERSTOP(mm_tiled_openmp);
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
#endif

    delete [] A;
    delete [] B;
    delete [] C;
}
