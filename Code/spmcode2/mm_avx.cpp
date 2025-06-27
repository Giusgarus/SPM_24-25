#include <random>       
#include <cstdint>      
#include <iostream>     
#include <immintrin.h>  // AVX intrinsics

#include "hpc_helpers.hpp"


void checkResult(float *C, float *Cc,
                 uint64_t m, uint64_t n) {
    for (uint64_t i = 0; i < m; i++)
        for (uint64_t j = 0; j < n; j++) {
            if (std::abs(C[i*n+j] - Cc[i*n+j]) > 1e-4f) {
                std::printf("Error %f, expected %f [%ld,%ld]\n", C[i*n+j], Cc[i*n+j], i, j);
                return;	
            }
        }
}

void init(float * data, uint64_t length) {

    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1, 1);

    for (uint64_t i = 0; i < length; i++)
        data[i] = density(engine);
}

void naive_mm(const float* A, const float* B, float* C,
			  uint64_t M, uint64_t L, uint64_t N) {
	for (uint64_t i = 0; i < M; i++)
        for (uint64_t j = 0; j < N; j++) {
            float accum = 0;
			for (uint64_t k = 0; k < L; k++)
                accum += A[i*L+k]*B[k*N+j];
			C[i*N+j] = accum;
		}
}

// This implementation uses AVX2 intrinsics and assumes N is a multiple of 8.
void naive_mm_avx(const float* A, const float* B, float* C,
				  uint64_t M, uint64_t L, uint64_t N) {

    for (uint64_t i = 0; i < M; ++i) {
        // Process columns of C in blocks of 8
        for (uint64_t j = 0; j < N; j += 8) {
            // Initialize accumulator vector to zero.
            __m256 X = _mm256_setzero_ps();

            for (uint64_t k = 0; k < L; ++k) {
                // Broadcast A[i][k] to all elements in the vector.
                __m256 Aik = _mm256_broadcast_ss(&A[i * L + k]);
                // Load 8 contiguous floats from B[k][j...j+7].
                __m256 BV  = _mm256_load_ps(&B[k * N + j]);
                // Fused multiply-add: cVec += aVal * bVec.
                X = _mm256_fmadd_ps(Aik, BV, X);
            }

            // Store the computed 8 values into C[i][j...j+7].
            _mm256_storeu_ps(&C[i * N + j], X);
        }
    }
}
#if 0
void looporder_mm_avx(const float* A, const float* B, float* C,
					  uint64_t M, uint64_t L, uint64_t N) {
    // Loop over i and k as before:
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t k = 0; k < L; k++) {
            // Broadcast A[i*l+k] to all lanes of a 256-bit vector (8 floats)
            __m256 a_ik = _mm256_set1_ps(A[i * L + k]);

            // Process the j-loop in chunks of 8 floats.
            for (uint64_t j = 0; j < N; j += 8) {
                // Load 8 floats from B[k*n+j] and C[i*n+j]
                __m256 b_vec = _mm256_loadu_ps(&B[k * n + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * n + j]);

                // Multiply and accumulate:
                // Using FMA if available:  c_vec += a_ik * b_vec
                c_vec = _mm256_fmadd_ps(a_ik, b_vec, c_vec);
                
                // Store the result back to C[i*n+j]
                _mm256_storeu_ps(&C[i * n + j], c_vec);
            }
        }
    }
}
#endif

int main (int argc, char *argv[]) {
	uint64_t def_m=10, def_n=11, def_l=12;
	if (argc != 1 && argc != 4) {
		std::printf("use: %s m n l\n", argv[0]);
		return -1;
	}
	if (argc > 1) {
		def_m = std::stol(argv[1]);
		def_n = std::stol(argv[2]);
		def_l = std::stol(argv[3]);
	}

    const uint64_t M = 1UL <<  def_m;
    const uint64_t L = 1UL <<  def_n;
    const uint64_t N = 1UL <<  def_l;

	// allocate aligned memory
    TIMERSTART(alloc_memory);
    auto A = static_cast<float*>(_mm_malloc(M*L*sizeof(float) , 32));
    auto B = static_cast<float*>(_mm_malloc(N*L*sizeof(float) , 32));
    auto C = static_cast<float*>(_mm_malloc(M*N*sizeof(float) , 32));
    auto Ccheck = static_cast<float*>(_mm_malloc(M*N*sizeof(float) , 32));
    TIMERSTOP(alloc_memory);

    TIMERSTART(init);
    init(A, M*L);
    init(B, L*N);
    TIMERSTOP(init);

    TIMERSTART(naive_mm);
    naive_mm(A, B, Ccheck, M, L, N);
    TIMERSTOP(naive_mm);

    TIMERSTART(naive_mm_avx);
    naive_mm_avx(A, B, C, M, L, N);
    TIMERSTOP(naive_mm_avx);

    //checkResult(C, Ccheck, M, N);	
    
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(Ccheck);
}

