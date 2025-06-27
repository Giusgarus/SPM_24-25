/*
 * group.cpp
 *
 * MPI Group Creation and Intra‐group Allreduce Example
 *
 * Description:
 *   - Initializes MPI and obtains global rank ('myrank') and size ('size') of MPI_COMM_WORLD.
 *   - Splits all processes into two groups:
 *       - "lower" group: ranks 0 through (size+1)/2−1  
 *       - "upper" group: ranks (size+1)/2 through size−1
 *   - Retrieves the MPI_Group of MPI_COMM_WORLD via MPI_Comm_group.
 *   - Uses MPI_Group_incl to build 'new_group' containing only the ranks in the lower or upper half, depending on 'myrank'.
 *   - Determines each process’s rank within 'new_group' with MPI_Group_rank.
 *   - Creates a new communicator 'new_comm' from 'new_group' via MPI_Comm_create.
 *   - Performs an MPI_Allreduce over 'new_comm' to sum the original world-ranks ('myrank') across each subgroup.
 *   - Each process prints its original rank, its rank within the new communicator, and the sum of original ranks in its subgroup.
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

	std::vector<int> lower;
	for(int i=0;i<(size+1)/2; ++i)
		lower.push_back(i);
	std::vector<int> upper;
	for(int i=(size+1)/2;i<size; ++i)
		upper.push_back(i);
		
	// Extract the group from the "universe" communicator
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	// Create a new group containing either the lower or the upper ranks
	MPI_Group new_group;
	if (myrank < (size+1)/2) {
		MPI_Group_incl(world_group, (size+1)/2, lower.data(), &new_group);
	} else {
		MPI_Group_incl(world_group, size/2, upper.data(), &new_group);
	}

	// this is my rank in new_group 
	int newrank;
	MPI_Group_rank(new_group, &newrank);
	
	// Create a new communicator from the group and the "universe" communicator
	MPI_Comm new_comm;	
	MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

	// summing the (old) ranks in my group 
	int sndbuf = myrank;
	int rcvbuf;
	MPI_Allreduce(&sndbuf, &rcvbuf, 1, MPI_INT, MPI_SUM, new_comm);

	std::printf("rank=%d, newrank=%d, rcvbuf=%d\n", myrank, newrank, rcvbuf);
	
	MPI_Finalize();
}
