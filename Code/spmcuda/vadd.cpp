//
// Sum of two arrays in CUDA 
//
// Compile with: 
//  nvcc -std=c++17 -I . -x cu -O3 vadd.cpp -o vadd
//


#include <iostream>
#include <hpc_helpers.hpp>

constexpr int THREADS_PER_BLK=128;  // Number of threads per block

__global__ void add( int *x, int *y, int *z, int N ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx < N ) {
        z[idx] = x[idx] + y[idx];
    }
}

void init(int *x,int *y, int N) {
	for(int i=0;i<N;++i){
		x[i]=i;
		y[i]=i/2;
	}
}

void checkResult(int *x, int *y, int *z, int N) {
    for (int i=0; i<N; i++) {
        if ( z[i] != x[i] + y[i] ) {
			std::printf("Error at index %d: x[%d]=%d, y[%d]=%d, z[%d]=%d\n",
						i, i, x[i], i, y[i], i, z[i]);
            break;
        }
    }
}

int main() {
	const int N = 1024*1024;
	
	// Allocate space on the host for the three vectors 
	int *x = new int[N];
	int *y = new int[N];
	int *z = new int[N];
	init(x,y,N);
	

    // Device pointers to data
    int *d_x, *d_y, *d_z;

	TIMERSTART(alloc_transfer);

    // Allocate space for device copies 
    cudaMalloc((void**)&d_x, N*sizeof(int)); CUERR;
    cudaMalloc((void**)&d_y, N*sizeof(int)); CUERR;
    cudaMalloc((void**)&d_z, N*sizeof(int)); CUERR;
    
    // Copy inputs to device
    cudaMemcpy(d_x, x, N*sizeof(int), H2D); CUERR;
    cudaMemcpy(d_y, y, N*sizeof(int), H2D); CUERR;
	TIMERSTOP(alloc_transfer);

	TIMERSTART(kernel);
    // Launch add kernel on the device
	int numBlocks = (N + THREADS_PER_BLK-1) / THREADS_PER_BLK;
    add<<<numBlocks, THREADS_PER_BLK>>>(d_x, d_y, d_z, N); CUERR;
	cudaDeviceSynchronize(); CUERR;
	TIMERSTOP(kernel);

	TIMERSTART(copy_back);
    // Copy result back to host
    cudaMemcpy(z, d_z, N*sizeof(int), D2H); CUERR;
	TIMERSTOP(copy_back);
	
	checkResult(x,y,z, N);

    delete [] x;
    delete [] y;
    delete [] z;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}
