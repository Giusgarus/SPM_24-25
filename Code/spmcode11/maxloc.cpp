//
// Example of MPI_Reduce using the MAXLOC operation
//
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	srandom(rank);

	// This corresponds to the predefined type MPI_2INT
    struct my2INT { 
        int val;
        int idx;
    } in, out;
	
    in.val = random()%100;
    in.idx = rank;

	std::printf("Process %d val %d\n", in.idx, in.val);
	
    MPI_Reduce(&in, &out, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // after the reduce, the maxloc values are in the root rank 
    if (!rank) {
		std::printf("The maximum is %d with idx %d\n", out.val, out.idx);
    }

    MPI_Finalize();
    return 0;
}

