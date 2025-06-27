#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <hpc_helpers.hpp>

void checkResult(const float *x, const float *y, const float a, int n,
				 const float *y2) {
	for(int i = 0; i<n; ++i) {
		float v = a * x[i] + y[i];
		if (std::abs(y2[i] - v) > 1e-4f) {
			std::printf("Error %f, expected %f [%d]\n", y2[i], v, i);
			return;
		}
	}
}

// SAXPY kernel: y[i] = a*x[i] + y[i]
__global__ void saxpy(const float* x, float* y, float a, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

void init(float *x, float *y, int N) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }	
}

int main() {
    const int N = 1 << 20; // 1 million elements
    const float a = 2.5f;

	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      std::printf("No CUDA devices found on this machine\n");
      return -1;
    }
    std::printf("CUDA Device Count: %d\n", deviceCount);

    // in case of multiple device, select one
	int device=0;
    cudaSetDevice(device); CUERR;
	
    // Allocate Unified Memory (Managed memory)
    float* x = nullptr;
    float* y = nullptr;
    cudaMallocManaged(&x, N * sizeof(float)); CUERR;
    cudaMallocManaged(&y, N * sizeof(float)); CUERR;

	init(x,y, N);

	// to check results
    float* y2 = new float[N];
	for(int i=0; i<N; ++i)
		y2[i] = y[i];				 
	
	TIMERSTART(saxpy);
#if 0
    // Provide memory advice to the CUDA runtime (we intend to access x,y)
    // The runtime can use this hint to optimize placement.
    cudaMemAdvise(x, N * sizeof(float), cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(y, N * sizeof(float), cudaMemAdviseSetPreferredLocation, device);

    // Prefetch x and y to the GPU to avoid on-demand page faults
    cudaMemPrefetchAsync(x, N * sizeof(float), device);
    cudaMemPrefetchAsync(y, N * sizeof(float), device);
#endif
	
    // Launch kernel
    int blockSize = 128;
    int gridSize  = (N + blockSize - 1) / blockSize;
    saxpy<<<gridSize, blockSize>>>(x, y, a, N); CUERR;	
    cudaDeviceSynchronize(); CUERR;

#if 0	
    // Prefetch data back to host
    cudaMemPrefetchAsync(y, N*sizeof(float), cudaCpuDeviceId);
#endif	
	checkResult(x,y2,a,N, y);
	TIMERSTOP(saxpy);

    // Free Unified Memory
    cudaFree(x);
    cudaFree(y);
	delete [] y2;
}
