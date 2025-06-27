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
	
    float* x = new float[N];
    float* y = new float[N];
	init (x,y, N);

	// to check results
    float* y2 = new float[N];
	for(int i=0; i<N; ++i)
		y2[i] = y[i];				 

	
	float *d_x, *d_y;
    // Allocate space for device copies 
    cudaMalloc((void**)&d_x, N * sizeof(float)); CUERR;
    cudaMalloc((void**)&d_y, N * sizeof(float)); CUERR;

	TIMERSTART(saxpy);	
	// Copy inputs to device
    cudaMemcpy(d_x, x, N*sizeof(float), H2D); CUERR;
    cudaMemcpy(d_y, y, N*sizeof(float), H2D); CUERR;
		
    // Launch kernel
    int blockSize = 128;
    int gridSize  = (N + blockSize - 1) / blockSize;
    saxpy<<<gridSize, blockSize>>>(d_x, d_y, a, N); CUERR;	
    cudaDeviceSynchronize(); CUERR;

    // Copy result back to host
    cudaMemcpy(y, d_y, N*sizeof(float), D2H); CUERR;
	
	checkResult(x,y2,a,N, y);
	TIMERSTOP(saxpy);

    // Free Unified Memory
    cudaFree(d_x);
    cudaFree(d_y);
	delete [] y2;
	delete [] x;
	delete [] y;
}
