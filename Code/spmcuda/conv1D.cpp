//
// 1D Convolution Using CUDA kernel and Shared Memory
//
// How to compile:
//  nvcc -std=c++17 -x cu -O3 conv1D.cpp -o conv1D
//

#include <iostream>
#include <cuda_runtime.h>
#include "hpc_helpers.hpp"

constexpr int THREADS_PER_BLK=128;  // Number of threads per block

// suboptimal CUDA Kernel using only Global memory
__global__ void naive_conv1D(int N, float *input, float *output) {

    // Compute global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Apply 1D convolution
	float sum = 0.0f;
	for (int i = 0; i < 3; ++i) {
		sum += input[idx + i];
	}
	output[idx] = sum / 3.0f; // Store the result
}

// CUDA Kernel using Shared Memory 
__global__ void conv1D(int N, float *input, float *output) {

    // Shared memory buffer with extra space for boundary elements
    __shared__ float shm[THREADS_PER_BLK + 2];

    // Compute global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global memory to shared memory
    shm[threadIdx.x] = input[idx];

    // Load boundary elements
    if (threadIdx.x < 2) 
        shm[THREADS_PER_BLK + threadIdx.x] = input[idx + THREADS_PER_BLK];

    // Barrier among all threads in the block to ensure
	// shared memory is updated
    __syncthreads();

    // Apply 1D convolution
	float sum = 0.0f;
	for (int i = 0; i < 3; ++i) {
		sum += shm[threadIdx.x + i]; // Use shared memory for fast access
	}
	output[idx] = sum / 3.0f; // Store the result
}


int main() {
    const int N= 1024*1024;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      std::printf("No CUDA devices found on this machine\n");
      return -1;
    }
    std::printf("CUDA Device Count: %d\n", deviceCount);

    // in case of multiple device, select one
    cudaSetDevice(0); CUERR;
    
	
    // Host arrays
    float *h_input  = new float[N+2];
    float *h_output = new float[N];
	
    // Initialize input data
    for (int i = 0; i < (N+2); i++) {
        h_input[i] = static_cast<float>(i + 1); 
    }

    // Device pointers
    float *d_input, *d_output;
    
    TIMERSTART(alloc_and_transfer);
    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, (N+2) * sizeof(float)); CUERR;
    cudaMalloc((void**)&d_output, N * sizeof(float)); CUERR;
    
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, (N+2) * sizeof(float), H2D); CUERR;
    TIMERSTOP(alloc_and_transfer);
	
    // Launch Kernel (Using enough blocks to process all elements)
    int numBlocks = (N + THREADS_PER_BLK - 1) / THREADS_PER_BLK;

    TIMERSTART(naive_conv1D);
    naive_conv1D<<<numBlocks, THREADS_PER_BLK>>>(N, d_input, d_output); CUERR;
    cudaDeviceSynchronize(); // wait for the kernel to finish
    TIMERSTOP(naive_conv1D);

    TIMERSTART(conv1D);
    conv1D<<<numBlocks, THREADS_PER_BLK>>>(N, d_input, d_output); CUERR;
    cudaDeviceSynchronize(); // wait for the kernel to finish
    TIMERSTOP(conv1D);

    // Copy results back to host
    TIMERSTART(copy_back);
    cudaMemcpy(h_output, d_output, N * sizeof(float), D2H); CUERR;
    TIMERSTOP(copy_back);
    TIMERSUM(alloc_and_transfer, copy_back);
#if 0
    TIMERSTART(conv1D);
    conv1D<<<numBlocks, THREADS_PER_BLK>>>(N, d_input, d_output); CUERR;
    cudaMemcpy(h_output, d_output, N * sizeof(float), D2H); CUERR;
    TIMERSTOP(conv1D);
    TIMERSUM(alloc_and_transfer, conv1D);
#endif

    // Print a few output values
    std::cout << "Output: ";
    for (int i = 0; i < 8; i++) {
      std::printf("%.1f ", h_output[i]);
    }
    std::cout << " ... ";
    for (int i = N-8; i < N; i++) {
      std::printf("%.1f ", h_output[i]);
    }
    std::cout << "\n";
    
   // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
   delete [] h_input;
   delete [] h_output;
   return 0;
}
