#include <iostream>
#include <immintrin.h>  
#include "hpc_helpers.hpp"


// AVX2 Optimized 1D Convolution
void conv1D_AVX2(int N, const float* input, float* output) {
    
    // Process 8 elements at a time using AVX2
    __m256 div_factor = _mm256_set1_ps(3.0f);  // Vector of 3.0f for division

	int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v0 = _mm256_load_ps(&input[i]);     // Load input[i] to input[i+7]
        __m256 v1 = _mm256_load_ps(&input[i + 1]); // Load input[i+1] to input[i+8]
        __m256 v2 = _mm256_load_ps(&input[i + 2]); // Load input[i+2] to input[i+9]

        __m256 sum = _mm256_add_ps(v0, v1);        // v0 + v1
        sum = _mm256_add_ps(sum, v2);              // (v0 + v1) + v2

        __m256 result = _mm256_div_ps(sum, div_factor); // sum / 3.0f
        _mm256_store_ps(&output[i], result);  // Store results in output
    }

    // Handle remaining elements (scalar computation)
    for (; i < N; i++)
        output[i] = (input[i] + input[i + 1] + input[i + 2]) / 3.0f;
}

int main() {
	// Define the size of the input array
	constexpr int N = 1024 * 1024; 


	// Allocate aligned memory for input and output
    float* input = (float*)aligned_alloc(32, (N + 2) * sizeof(float)); // N+2 to avoid boundary issues
    float* output = (float*)aligned_alloc(32, N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N + 2; i++) {
        input[i] = static_cast<float>(i + 1);
    }

	TIMERSTART(conv1D);
    conv1D_AVX2(N, input, output);
	TIMERSTOP(conv1D);

    // Print output for verification
    std::cout << "Output: ";
    for (int i = 0; i < 8; i++) {
        printf("%.1f ", output[i]);
    }
    std::cout << " ... ";
    for (int i = N - 8; i < N; i++) {
        printf("%.1f ", output[i]);
    }
    std::cout << "\n";

    // Free allocated memory
    free(input);
    free(output);

    return 0;
}
