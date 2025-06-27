/*
 * mm_2d.cpp
 *
 * Parallel 2D-blocked matrix multiplication with MPI
 *
 * Description:
 *   This program computes C = alpha*A*B in parallel by decomposing the
 *   global m * k and k * n matrices into sqrt(P) * sqrt(P) blocks across P MPI processes.
 *   Process 0:
 *     - Reads the problem size (m, k, n) and scalar alpha via readParams()
 *     - Broadcasts these parameters to all ranks using a derived struct type
 *     - Allocates and initializes the full matrices A (m*k) and B (k*n)
 *     - Scatters row-blocks of A and column-blocks of B to each process
 *       using MPI_Type_contiguous and MPI_Type_vector, respectively
 *   All processes:
 *     - Receive their local submatrices myA and myB
 *     - Perform the local matrix multiplication myC = alpha * myA *myB
 *     - Send back myC to process 0
 *   Process 0:
 *     - Gathers all subblocks into the full C matrix using a 2D vector type
 *     - Writes C to "out2D.txt"
 *
 *
 * Preconditions:
 *   - P must be a perfect square
 *   - m and n must be divisible by sqrt(P)  
 */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include "mpi.h"

struct params{
  int m, k, n;
  float alpha;
};

void readParams(params* p){
	// reading here from a file
	
	p->m = 1024;
	p->k = 1024;
	p->n = 1024;
	p->alpha = 1.5;
}

void readInput(int m, int k, int n, float *A, float *B){
    for(int i=0; i<m; i++)
        for(int j=0; j<k; j++)
            A[i*k+j] = (i+j) % 2;

    for(int i=0; i<k; i++)
        for(int j=0; j<n; j++)
            B[i*n+j] = (i+j) % 2;
}

void printOutput(int rows, int cols, float *data){

	FILE *fp = fopen("out2D.txt", "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file out2D.txt could not be opened" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
        	fprintf(fp, "%lf ", data[i*cols+j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main (int argc, char *argv[]){
	// Initialize MPI
	MPI_Init(&argc, &argv);
	int numP;
	int myId;
	// Get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &numP);
	int gridDim = sqrt(numP);

	if(gridDim*gridDim != numP){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: the number of processes must be square" << std::endl;

		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	params p;
	MPI_Aint base;
	MPI_Aint disp[2];
	MPI_Get_address(&p,         &base);
	MPI_Get_address(&p.m,       &disp[0]);
	MPI_Get_address(&p.alpha,   &disp[1]);
	disp[0] -= base;
	disp[1] -= base;
	int blocklens[2]  = {3,1};
	MPI_Datatype types2[2] = {MPI_INT, MPI_FLOAT};

	// Create the datatype for the parameters
	MPI_Datatype paramsType;
	MPI_Type_create_struct(2, blocklens, disp, types2, &paramsType);
	MPI_Type_commit(&paramsType);

	if(!myId){
		// Process 0 reads the parameters from a configuration file
		readParams(&p);
	}
	// Broadcast of all the parameters using one message with a struct
	MPI_Bcast(&p, 1, paramsType, 0, MPI_COMM_WORLD);

	if((p.m < 1) || (p.n < 1) || (p.k<1)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm', 'k' and 'n' must be higher than 0" << std::endl;
		
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	if((p.m%gridDim) || (p.n%gridDim)){
		// Only the first process prints the output message
		if(!myId)
			std::cout << "ERROR: 'm', 'n' must be multiple of the grid dimensions" << std::endl;

		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	float *A=nullptr;
	float *B=nullptr;
	float *C=nullptr;
	float *myA;
	float *myB;
	float *myC;
	int blockRows = p.m/gridDim;
	int blockCols = p.n/gridDim;
	
	// Only one process reads the data from the files
	if(!myId){
		A = new float[p.m*p.k];
		B = new float[p.k*p.n];
		C = new float[p.m*p.n];
		readInput(p.m, p.k, p.n, A, B);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	// Create the datatype for a block of rows of A
	MPI_Datatype rowsType;
	MPI_Type_contiguous(blockRows * p.k, MPI_FLOAT, &rowsType);
	MPI_Type_commit(&rowsType);

	myA = new float[blockRows * p.k];
	std::vector<int> sendcounts_A(numP, 1), displs_A(numP);
	for (int r = 0; r < numP; ++r) {
		int i = r / gridDim;   // logical row
		displs_A[r] = i;       // offset (multiple of rowsType)
	}
	MPI_Scatterv(A, sendcounts_A.data(),  displs_A.data(),  // 0,0,1,1 for gridDim=2
				 rowsType,       
				 myA, blockRows * p.k, MPI_FLOAT,
				 0, MPI_COMM_WORLD);
	
	// Create the datatype for a block of columns of B
	MPI_Datatype colsType;
	// fields: count, blocklen, stride, type, newtype
	MPI_Type_vector(p.k, blockCols, p.n, MPI_FLOAT, &colsType);
    MPI_Aint lb, old_ext;
    MPI_Type_get_extent(colsType, &lb, &old_ext);
	MPI_Datatype colsTypeResized;
    MPI_Type_create_resized(colsType, lb, blockCols * sizeof(float), &colsTypeResized);
    MPI_Type_commit(&colsTypeResized);
	MPI_Type_free(&colsType);

	myB = new float[p.k * blockCols];
	// prepare Scatterv for B: each rank receives block column j = rank % gridDim
    std::vector<int> sendcounts_B(numP, 1), displs_B(numP);
    for(int r=0; r<numP; ++r) {
        int j = r % gridDim;
        displs_B[r] = j;  // in units of colsTypeResized extents
    }
    MPI_Scatterv(B, sendcounts_B.data(), displs_B.data(), colsTypeResized,
				 myB, p.k * blockCols, MPI_FLOAT,
				 0, MPI_COMM_WORLD);

	// Array for the chunk of data to work
	myC = new float[blockRows*blockCols];

	// The multiplication of the submatrices
	for(int i=0; i<blockRows; i++)
		for(int j=0; j<blockCols; j++){
			myC[i*blockCols+j] = 0.0;
			for(int l=0; l<p.k; l++)
				myC[i*blockCols+j] += p.alpha*myA[i*p.k+l]*myB[l*blockCols+j];
		}

	MPI_Datatype block2DType, block2DTypeResized;
	MPI_Type_vector(blockRows, blockCols, p.n, MPI_FLOAT, &block2DType);
	MPI_Type_create_resized(block2DType,
							0,                          /* LB   */
							blockCols*sizeof(float),    /* extent = larghezza */
							&block2DTypeResized);
	MPI_Type_commit(&block2DTypeResized);
	MPI_Type_free(&block2DType);       

	std::vector<int> recvcounts_C(numP, 1), displs_C(numP);
	for (int r = 0; r < numP; ++r) {
		int i = r / gridDim;       // row coordinate
		int j = r % gridDim;       // col coordinate
		displs_C[r] = i * blockRows * gridDim + j;
	}

	float* rootBuf = (!myId) ? C : nullptr; 
	MPI_Gatherv(myC, blockRows*blockCols, MPI_FLOAT,
				rootBuf, recvcounts_C.data(), displs_C.data(),
				block2DTypeResized,         
				0, MPI_COMM_WORLD);
	
	
	// Measure the current time and print by process 0
	double end = MPI_Wtime();
	
	if(!myId){
    		std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    		printOutput(p.m, p.n, C);
		delete [] A;
		delete [] B;
		delete [] C;
	}

	// Delete the types and arrays
	MPI_Type_free(&rowsType);
	MPI_Type_free(&colsTypeResized);
	MPI_Type_free(&block2DTypeResized);

	delete [] myA;
	delete [] myB;
	delete [] myC;

	// Terminate MPI
	MPI_Finalize();
	return 0;
}
