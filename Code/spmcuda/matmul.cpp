//
// Tile-based GEMM 
//
//
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "hpc_helpers.hpp"

// Define block (tile) size: 16 or 32 are common choices
constexpr int BLOCK_SIZE=32;

//  Compute C = A * B
//    A: (M x L)   
//    B: (L x N) 
//    C: (M x N)
//
__global__ void mmTiled(const float *A, const float *B, float *C,
						int M, int N, int L) {

    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    // compute the row and column of the C element this thread will produce
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    // Each thread accumulates one element of C in 'c'
    float c = 0.0f;

    // Loop over tiles along the L dimension
    int numTiles = (L + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Global memory indices to load from
        const int tcolA = t * BLOCK_SIZE + threadIdx.x;  // for A
        const int trowB = t * BLOCK_SIZE + threadIdx.y;  // for B
        // Load tile from A into shared memory
        if (row < M && tcolA < L) {
            sA[threadIdx.y][threadIdx.x] = A[row * L + tcolA];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load tile from B into shared memory
        if (col < N && trowB < L) {
            sB[threadIdx.y][threadIdx.x] = B[trowB * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Make sure the entire block has loaded the tile before computing
        __syncthreads();
        // Multiply the loaded tiles
        for (int i = 0; i < BLOCK_SIZE; i++) {
            c += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        // Ensure all threads are done reading sA/sB before loading next tile
        __syncthreads();
    }
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = c;
    }
}

__global__ void mm(const float *A, const float *B, float *C,
				   int M, int N, int L) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if (row < M && col < N) {
		float c = 0.0f;
		for(int k=0;k<L;++k)
			c += A[row*L +k] * B[k*N + col];
		
		C[row * N + col] = c;    
	}
}



void checkResult(float *C, float *Cc,
                 uint64_t M, uint64_t N) {
    for (uint64_t i = 0; i < M; i++)
        for (uint64_t j = 0; j < N; j++) {
            if (std::abs(C[i*N+j] - Cc[i*N+j]) > 1e-3f) {
                std::printf("Error %f, expected %f [%ld,%ld]\n", C[i*N+j], Cc[i*N+j], i, j);
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


int main(int argc, char *argv[]) {
	size_t def_m=15, def_n=15, def_l=5;
	if (argc != 1 && argc != 4) {
		std::printf("use: %s m n l\n", argv[0]);
		return -1;
	}
	if (argc > 1) {
		def_m = std::stol(argv[1]);
		def_n = std::stol(argv[2]);
		def_l = std::stol(argv[3]);
	}

    // matrix shapes
    const size_t M = 1 << def_m;
    const size_t N = 1 << def_n;
    const size_t L = 1 << def_l;
  
    // allocate device memory
    size_t sizeA = M * L * sizeof(float);
    size_t sizeB = L * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA); CUERR;
    cudaMalloc(&dB, sizeB); CUERR;
    cudaMalloc(&dC, sizeC); CUERR;

	float *hA, *hB, *hC;
	hA = new float[M*L];
	hB = new float[L*N];
	hC = new float[M*N];
	init(hA, M*L);
	init(hB, L*N);

    cudaMemcpy(dA, hA, sizeA, H2D); CUERR;
	cudaMemcpy(dB, hB, sizeB, H2D); CUERR;
	    
    // compute grid dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);
	
	TIMERSTART(mm_tiled);
    // launch the kernel
    mmTiled<<<grid, block>>>(dA, dB, dC, M, N, L); CUERR;
    // wait for kernel to complete
    cudaDeviceSynchronize(); CUERR;
	TIMERSTOP(mm_tiled);

	// copy back C matrix
    cudaMemcpy(hC, dC, sizeC, D2H); CUERR;

	TIMERSTART(mm_naive);
	mm<<<grid, block>>>(dA, dB, dC, M, N, L); CUERR;
	cudaDeviceSynchronize(); CUERR;
	TIMERSTOP(mm_naive);

	float *tmpC = new float[M*N];
    cudaMemcpy(tmpC, dC, sizeC, D2H); CUERR;

	checkResult(hC, tmpC, M, N);
	delete [] tmpC;

    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC); 
	delete [] hA;
	delete [] hB;
	delete [] hC;	   
}
