/*
 * struct.cpp
 *
 * MPI Derived Struct Datatype Example: Sending a Particle Structure
 *
 * Demonstrates how to handle heterogeneous struct layouts without manual
 * packing, by leveraging 'MPI_Type_create_struct' and address arithmetic.
 * Ensures portability across architectures with different padding/alignment.

 * Description:
 *   - Defines a C++ 'Particle' struct containing two 'double' fields
 *     followed by one 'int' field
 *   - Uses 'MPI_Get_address' to compute byte displacements of 'x' and 'charge'
 *     relative to the struct base to capture any padding and alignment.
 *   - Builds a custom MPI datatype 'particle_type' via 'MPI_Type_create_struct'
 *     with block lengths {2,1} and types {MPI_DOUBLE, MPI_INT}, then commits it.
 *   - On rank 0:
 *       - Initializes a 'Particle send_particle' with example values.
 *       - Sends it to **rank 1** using 'MPI_Send' and the derived datatype.
 *       - Prints confirmation of the sent values.
 *   - On rank 1:
 *       - Receives into a local 'Particle recv_particle' via 'MPI_Recv'
 *         using 'particle_type'.
 *       - Prints the received values
 */

#include <cstdio>
#include <vector>
#include <mpi.h>

struct Particle {
	double x; 
	double y;
	int    charge;
};


int main(int argc, char* argv[]) {
	int rank;
	int size;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
    // The internal Particle types are {MPI_DOUBLE, MPI_INT}
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    // a block of two double and one int
    int block_lengths[2] = {2, 1};

	Particle p;
    MPI_Aint displacements[2];
    MPI_Aint base_address;
    MPI_Get_address(&p,        &base_address);
    MPI_Get_address(&p.x,      &displacements[0]);
    MPI_Get_address(&p.charge, &displacements[1]);
    // convert the addresses to be relative to the base address
    displacements[0] -= base_address;
    displacements[1] -= base_address;

    MPI_Datatype particle_type;
    MPI_Type_create_struct(2, block_lengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    // now use the new type to send/receive a particle
    if(rank == 0) {
        Particle send_particle;
        send_particle.x = 1.0; send_particle.y = -2.5; send_particle.charge = 42;
        MPI_Send(&send_particle, 1, particle_type, 1, 0, MPI_COMM_WORLD);
		std::printf("rank 0 sent Particle{%f, %f, %d} to rank 1\n",
					send_particle.x, send_particle.y, send_particle.charge);
    } else
		if(rank == 1) {
        Particle recv_particle;
        MPI_Recv(&recv_particle, 1, particle_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::printf("rank 1 received Particle{%f, %f, %d} from rank 0\n",
					recv_particle.x, recv_particle.y, recv_particle.charge);
    }

    MPI_Type_free(&particle_type);
	MPI_Finalize();
}
	
