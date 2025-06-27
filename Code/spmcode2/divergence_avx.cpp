#include <immintrin.h>
#include <iostream>
#include <vector>
#include "hpc_helpers.hpp"


// Scalar version with branching (divergence)
void transform_scalar(const float* input, float* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (input[i] < 0.0f)
            output[i] = input[i] * 2.0f;
        else
            output[i] = input[i] / 2.0f;
    }
}

// AVX version without predication: it loads/stores vectors but processes
// each element with scalar branching
void transform_avx_branch(const float* input, float* output, size_t n) {
    size_t i = 0;
    // Process blocks of 8 floats using AVX load/store
    for (; i + 8 <= n; i += 8) {
        // Load 8 floats into an AVX register
        __m256 v = _mm256_loadu_ps(&input[i]);
        
        // Store the vector into a temporary array
        float temp[8];
        _mm256_storeu_ps(temp, v);
        
        // Process each element with scalar branching
		// could be vectorized by the compiler (divergence may occur here)
        for (int j = 0; j < 8; ++j) {
            temp[j] = (temp[j] < 0.0f) ? (temp[j] * 2.0f) : (temp[j] / 2.0f);
        }
        
        // Load the processed values back into an AVX register and store them to output
        v = _mm256_loadu_ps(temp);
        _mm256_storeu_ps(&output[i], v);
    }
    
    // Process any remaining elements that don't fit in a full AVX register
    for (; i < n; ++i) {
        if (input[i] < 0.0f)
            output[i] = input[i] * 2.0f;
        else
            output[i] = input[i] / 2.0f;
    }
}


// AVX version using predication (branch-free)
void transform_avx(const float* input, float* output, size_t n) {
    size_t i = 0;
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 two  = _mm256_set1_ps(2.0f);

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&input[i]);
        __m256 mulRes = _mm256_mul_ps(v, two);
        __m256 divRes = _mm256_div_ps(v, two);
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_LT_OS);
        __m256 blended = _mm256_blendv_ps(divRes, mulRes, mask);
        _mm256_storeu_ps(&output[i], blended);
    }
    // Process any remaining elements
    for (; i < n; ++i) {
        output[i] = (input[i] < 0.0f) ? input[i] * 2.0f : input[i] / 2.0f;
    }
}

int main() {
    const size_t n = 10000000; // vector elements
    std::vector<float> input(n);
    std::vector<float> output_scalar(n);
    std::vector<float> output_avx(n);
	std::vector<float> output_avx2(n);

    // Initialize input with random data, ensuring a mix of negative and positive values.
    for (size_t i = 0; i < n; ++i) {
        input[i] = (rand() % 200 - 100) / 10.0f; // values between -10 and 10
    }

	TIMERSTART(scalar);
    transform_scalar(input.data(), output_scalar.data(), n);
    TIMERSTOP(scalar);

	TIMERSTART(avx_branch);
    transform_avx_branch(input.data(), output_avx2.data(), n);
	TIMERSTOP(avx_branch);

	TIMERSTART(avx_predication);
    transform_avx(input.data(), output_avx.data(), n);
	TIMERSTOP(avx_predication);

    // Verify correctness
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(output_scalar[i] - output_avx[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << "\n";
            break;
        }
    }
    // Verify correctness
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(output_avx2[i] - output_avx[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << "\n";
            break;
        }
    }
	
    return 0;
}
