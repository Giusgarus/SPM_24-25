#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>
#include <immintrin.h>
#include <numeric>

void softmax_avx(const float *input, float *output, size_t K, int div = 0)
{
	int par_val = 8; // Number of elements processed in parallel using AVX
	// Initialize the maximum value to negative infinity
	__m256 max = _mm256_loadu_ps(input); // Load the first 8 elements of the input array

	size_t i = 8;
	// Find the maximum value in the input array using AVX
	for (; i + (par_val - 1) < K; i += par_val)
	{
		__m256 vec = _mm256_loadu_ps(&input[i]); // Load the next 8 elements
		max = _mm256_max_ps(max, vec); // Compute the element-wise maximum
	}

	// Scalar reduction to find the actual maximum value
	float max_arr[par_val];
	_mm256_storeu_ps(max_arr, max); // Store the max values into an array
	float max_final = max_arr[0];
	for (int j = 1; j < par_val; ++j)
	{
		if (max_arr[j] > max_final)
			max_final = max_arr[j]; // Find the maximum value in the array
	}
	for (; i < K; ++i)
	{
		if (input[i] > max_final)
			max_final = input[i]; // Check the remaining elements
	}

	// Calculate e^x and the total sum
	__m256 sum_sigma = _mm256_setzero_ps(); // Initialize sum to zero
	__m256 max_final_vec = _mm256_set1_ps(max_final); // Broadcast max_final to all elements
	i = 0;
	for (; i + (par_val - 1) < K; i += par_val)
	{
		__m256 sigma = _mm256_loadu_ps(&input[i]); // Load the next 8 elements
		sigma = _mm256_sub_ps(sigma, max_final_vec); // Subtract max_final from each element
		sigma = exp256_ps(sigma); // Compute the exponential of each element
		_mm256_storeu_ps(&output[i], sigma); // Store the result in the output array
		sum_sigma = _mm256_add_ps(sum_sigma, sigma); // Accumulate the sum
	}

	// Scalar reduction of the sum
	float arr_sum[par_val];
	_mm256_storeu_ps(arr_sum, sum_sigma); // Store the sum values into an array
	float sum_final = arr_sum[0];
	for (int j = 1; j < par_val; ++j)
		sum_final += arr_sum[j]; // Compute the total sum
	for (; i < K; ++i)
	{
		output[i] = std::exp(input[i] - max_final); // Compute the exponential for remaining elements
		sum_final += output[i]; // Accumulate the sum
	}

	if (div == 0)
	{
		// Normalization using multiplication
		__m256 sum_inv = _mm256_set1_ps(1.0f / sum_final); // Compute the inverse of the sum
		i = 0;
		for (; i + (par_val - 1) < K; i += par_val)
		{
			__m256 vec = _mm256_loadu_ps(&output[i]); // Load the next 8 elements
			vec = _mm256_mul_ps(vec, sum_inv); // Multiply each element by the inverse of the sum
			_mm256_storeu_ps(&output[i], vec); // Store the result in the output array
		}
		for (; i < K; ++i)
		{
			output[i] /= sum_final; // Normalize the remaining elements
		}
	}
	else
	{
		// Normalization using division
		__m256 sum_final_vec = _mm256_set1_ps(sum_final); // Broadcast sum_final to all elements
		i = 0;
		for (; i + (par_val - 1) < K; i += par_val)
		{
			__m256 vec = _mm256_loadu_ps(&output[i]); // Load the next 8 elements
			vec = _mm256_div_ps(vec, sum_final_vec); // Divide each element by the sum
			_mm256_storeu_ps(&output[i], vec); // Store the result in the output array
		}
		for (; i < K; ++i)
		{
			output[i] /= sum_final; // Normalize the remaining elements
		}
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

	TIMERSTART(softime_avx);
	softmax_avx(input.data(), output.data(), K);
	TIMERSTOP(softime_avx);

	// print the results on the standard output
	if (print)
	{
		printResult(output, K);
	}
}
