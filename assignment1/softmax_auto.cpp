#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <hpc_helpers.hpp>
#include <numeric>

void softmax_auto(const float *input, float *output, size_t K)
{
	// Find the maximum value
	float max_val1 = input[0], max_val2 = input[1];
	float max_val3 = input[2], max_val4 = input[3];

	size_t i = 4;
#pragma GCC ivdep // Indicate to the compiler that there are no dependencies in the loop
	for (; i + 3 < K; i += 4)
	{
		max_val1 = std::max(max_val1, input[i]);    // Find max in the first element of the group
		max_val2 = std::max(max_val2, input[i + 1]); // Find max in the second element of the group
		max_val3 = std::max(max_val3, input[i + 2]); // Find max in the third element of the group
		max_val4 = std::max(max_val4, input[i + 3]); // Find max in the fourth element of the group
	}
	float max_val = std::max(max_val1, std::max(max_val2, std::max(max_val3, max_val4))); // Find the overall max value

#pragma GCC ivdep
	for (; i < K; ++i)
	{
		if (input[i] > max_val)
			max_val = input[i]; // Update max_val if current element is greater
	}

	// Compute exponentials and their sum
	float sum = 0.0f;
#pragma GCC ivdep
// #pragma GCC unroll 8  // Suggest unrolling the loop to optimize vectorization
	for (size_t i = 0; i < K; ++i)
	{
		output[i] = std::exp(input[i] - max_val); // Compute the exponential of the input element minus max_val
		sum += output[i];                          // Accumulate the sum of exponentials
	}

	// Normalization
	float sum_inv = 1.0f / sum; // Compute the inverse of the sum
#pragma GCC ivdep
// #pragma GCC unroll 8
	for (size_t i = 0; i < K; ++i)
	{
		output[i] *= sum_inv; // Normalize each element by multiplying with the inverse of the sum
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

	TIMERSTART(softime_auto);
	softmax_auto(input.data(), output.data(), K);
	TIMERSTOP(softime_auto);

	// print the results on the standard output
	if (print)
	{
		printResult(output, K);
	}
}
