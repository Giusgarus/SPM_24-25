#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>
#include <immintrin.h>

// Function to reduce the maximum value in an AVX vector
inline float _mm256_hmax_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);                                // Extract the first 4 floats
    __m128 vhigh = _mm256_extractf128_ps(v, 1);                             // Extract the last 4 floats
    vlow = _mm_max_ps(vlow, vhigh);                                         // Compare the two halves
    vlow = _mm_max_ps(vlow, _mm_permute_ps(vlow, _MM_SHUFFLE(2, 3, 0, 1))); // Shuffle and compare
    vlow = _mm_max_ps(vlow, _mm_permute_ps(vlow, _MM_SHUFFLE(1, 0, 3, 2))); // Shuffle and compare
    return _mm_cvtss_f32(vlow);                                             // Extract the maximum
}

// Function to reduce the sum in an AVX vector
inline float _mm256_hadd_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);                                // Extract the first 4 floats
    __m128 vhigh = _mm256_extractf128_ps(v, 1);                             // Extract the last 4 floats
    vlow = _mm_add_ps(vlow, vhigh);                                         // Add the two halves
    vlow = _mm_add_ps(vlow, _mm_permute_ps(vlow, _MM_SHUFFLE(2, 3, 0, 1))); // Shuffle and add
    vlow = _mm_add_ps(vlow, _mm_permute_ps(vlow, _MM_SHUFFLE(1, 0, 3, 2))); // Shuffle and add
    return _mm_cvtss_f32(vlow);                                             // Extract the sum
}


void softmax_avx(const float *input, float *output, size_t K, int div = 0)
{
    int par_val = 8; // Number of parallel values processed by AVX
    __m256 max_vec = _mm256_loadu_ps(input); // Load the first 8 floats from input

    // Find the maximum value
    size_t i = 8;
    for (; i + par_val <= K; i += par_val)
    {
        __m256 vec = _mm256_loadu_ps(&input[i]); // Load 8 floats from input
        max_vec = _mm256_max_ps(max_vec, vec);   // Update max vector
    }
    float max_final = _mm256_hmax_ps(max_vec); // Reduce to find the maximum value
    for (; i < K; ++i)
    {
        if (input[i] > max_final)
            max_final = input[i]; // Handle remaining elements
    }

    // Compute e^x and the sum
    __m256 sum_sigma = _mm256_setzero_ps();           // Initialize sum vector
    __m256 max_final_vec = _mm256_set1_ps(max_final); // Broadcast max value
    i = 0;
    for (; i + par_val <= K; i += par_val)
    {
        __m256 vec = _mm256_loadu_ps(&input[i]);   // Load 8 floats from input
        vec = _mm256_sub_ps(vec, max_final_vec);   // Subtract max value
        vec = exp256_ps(vec);                      // Compute e^x
        _mm256_storeu_ps(&output[i], vec);         // Store result in output
        sum_sigma = _mm256_add_ps(sum_sigma, vec); // Update sum vector
    }

    float sum_final = _mm256_hadd_ps(sum_sigma); // Reduce to find the sum
    for (; i < K; ++i)
    {
        output[i] = std::exp(input[i] - max_final); // Handle remaining elements
        sum_final += output[i];                     // Update sum
    }

    // Normalization
    if (div == 0)
    {
        __m256 sum_inv = _mm256_set1_ps(1.0f / sum_final); // Compute reciprocal of sum
        i = 0;
        for (; i + par_val <= K; i += par_val)
        {
            __m256 vec = _mm256_loadu_ps(&output[i]); // Load 8 floats from output
            vec = _mm256_mul_ps(vec, sum_inv);        // Normalize
            _mm256_storeu_ps(&output[i], vec);        // Store result in output
        }
    }
    else
    {
        __m256 sum_final_vec = _mm256_set1_ps(sum_final); // Broadcast sum value
        i = 0;
        for (; i + par_val <= K; i += par_val)
        {
            __m256 vec = _mm256_loadu_ps(&output[i]); // Load 8 floats from output
            vec = _mm256_div_ps(vec, sum_final_vec);  // Normalize
            _mm256_storeu_ps(&output[i], vec);        // Store result in output
        }
    }
    for (; i < K; ++i)
    {
        output[i] /= sum_final; // Handle remaining elements
    }
}

std::vector<float> generate_random_input(size_t K, float min = -1.0f, float max = 1.0f)
{
    std::vector<float> input(K);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(5489); // fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < K; ++i)
    {
        input[i] = dis(gen);
    }
    return input;
}

void printResult(std::vector<float> &v, size_t K)
{
    for (size_t i = 0; i < K; ++i)
    {
        std::fprintf(stderr, "%f\n", v[i]);
    }
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        std::printf("use: %s K [1]\n", argv[0]);
        return 0;
    }
    size_t K = 0;
    if (argc >= 2)
    {
        K = std::stol(argv[1]);
    }
    bool print = false;
    if (argc == 3)
    {
        print = true;
    }

    std::vector<float> input = generate_random_input(K);
    std::vector<float> output(K);

    TIMERSTART(softime2_avx);
    softmax_avx(input.data(), output.data(), K);
    TIMERSTOP(softime2_avx);

    // print the results on the standard output
    if (print)
    {
        printResult(output, K);
    }
}
