/*
 * split.cpp
 *
 * MPI_Comm_split Example: Grouping Processes and Intra-group Reduction
 *
 * Description:
 *   - Initializes MPI and obtains the global rank (myrank) and size (numtasks).
 *   - Computes a 'color' for each process as (2*myrank)/numtasks, splitting all ranks into two groups:
 *       - color = 0 for the first half of ranks
 *       - color = 1 for the second half
 *     Uses 'key = myrank' to preserve ordering within each new communicator.
 *   - Calls MPI_Comm_split to create a new communicator (new_comm) for each group.
 *   - Within new_comm, each process determines its local newrank.
 *   - Performs an MPI_Allreduce over new_comm to sum the original myrank values
 *     across all processes in the same color group, storing the result in `rcvbuf`.
 *   - Each process prints its original rank, its newrank in the split communicator,
 *     and the sum of original ranks in its group.
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

	int key   = myrank;
	int color = (2*myrank) / size;

	MPI_Comm new_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);

	// this is my rank in new_comm
	int newrank;
	MPI_Comm_rank(new_comm, &newrank);
	
	// summing the (old) ranks in my group 
	int sndbuf = myrank;
	int rcvbuf;
	MPI_Allreduce(&sndbuf, &rcvbuf, 1, MPI_INT, MPI_SUM, new_comm);

	std::printf("rank=%d, newrank=%d, rcvbuf=%d\n", myrank, newrank, rcvbuf);
	
	MPI_Finalize();
}
