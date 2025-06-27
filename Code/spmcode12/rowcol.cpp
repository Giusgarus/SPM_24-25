/*
 * rowcol.cpp
 *
 * MPI Derived Datatypes: Sending Rows and Columns Example
 *
 * Description:
 *   - Initializes MPI and obtains each process's rank ('myrank') and total size ('size'). 
 *   - Defines two custom MPI datatypes on all ranks:
 *       - 'row_type': a contiguous block of 'size' integers (one full row).  
 *       . 'col_type': a strided vector of 'size' integers with stride 'size' (one full column).  
 *   - On rank 0
 *       1. Allocates and initializes a 'size * size' integer matrix 'data' with unique values.  
 *       3. Sends to each process 'p' (for 'p = 1...size-1'):
 *           - The 'p'-th row using 'row_type' with tag 100.  
 *           - The 'p'-th column using 'col_type' with tag 101.  
 *   - On ranks 1...size-1
 *       1. Receives the row (into 'row' array) and column (into 'col' array) from rank 0.  
 *       2. Prints the received row and column.  
 *
 */


#include <cstdio>
#include <vector>
#include <mpi.h>

int main(int argc, char* argv[]) {
	int myrank;
	int size;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// two new data types
	MPI_Datatype row_type;
	MPI_Datatype col_type;
   	
	// the row_type is contiguous
	MPI_Type_contiguous(size, MPI_INT, &row_type);
	// in the col_type each element has a stride of size 
	MPI_Type_vector(size, 1, size, MPI_INT, &col_type);
	MPI_Type_commit(&row_type);
	MPI_Type_commit(&col_type);
	
	if (myrank == 0) {  // root 
		int *data = new int[size*size];
		
		// initialize data
		for(int i=0;i<size;++i)
			for(int j=0;j<size;++j)
				data[i*size+j] = 1+ j + size*i;

		std::printf("rank 0:\n");
		for(int i=0;i<size;++i) {
			for(int j=0;j<size;++j)
				std::printf("%d, ", data[i*size + j]);
			std::printf("\n");
		}
		std::printf("\n");

		
		for(int p=1;p<size;++p) {
			MPI_Send(&data[p*size], 1, row_type, p, 100, MPI_COMM_WORLD);
			MPI_Send(&data[p], 1, col_type, p, 101, MPI_COMM_WORLD);
		}
	} else {
		std::vector<int> row(size);
		std::vector<int> col(size);
		
		MPI_Recv(row.data(), size, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(col.data(), size, MPI_INT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		std::printf("rank %d:\n", myrank);
		std::printf(" row: ");
		for(int i=0;i<size;++i) std::printf("%d, ", row[i]);
		std::printf("\n");
		std::printf(" col: ");
		for(int i=0;i<size;++i) std::printf("%d, ", col[i]);
		std::printf("\n");
	}

	MPI_Type_free(&row_type);
	MPI_Type_free(&col_type);

	MPI_Finalize();
}
