// Run with:    mpirun -n <num_procs> ./power_iteration <matrix_size> <max_iters> <tollerance>
// Example:     mpirun -np 4 ./power_iter 800 1000 1e-8
//
// A simple distributed-memory implementation of the power iteration algorithm.
// The matrix A is defined implicitly by the symmetric formula
//    A(i,j) = 1 / (1 + |i-j|)  (the "inverse distance" matrix)
// This avoids storing the full NxN matrix in memory.
//
// The program approximates the dominant eigenvalue (lambda_max) and eigenvector (v)
// until ||v_{k} − v_{k-1}|| < tollerance or max_iters iterations are reached.
//
#include <mpi.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <random>

inline double Aij(int i, int j) {
    return 1.0 / (1.0 + std::abs(i - j));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int    N          = (argc > 1) ? std::atoi(argv[1]) : 1000;  
    const int    max_iters  = (argc > 2) ? std::atoi(argv[2]) : 1000;  
    const double tollerance = (argc > 3) ? std::atof(argv[3]) : 1e-8;  

    if (rank == 0) {
        std::cout << "Power iteration on inverse-distance matrix of size "
                  << N << "x" << N << " (up to " << max_iters
                  << " iterations, tollerance=" << tollerance << ")\n";
    }

    // Row distribution (block partitioning)
    const int rows_per_proc = N / size;
    const int remainder     = N % size;
    const int local_rows    = rows_per_proc + (rank < remainder ? 1 : 0);
    const int row_start     = rank * rows_per_proc + std::min(rank, remainder);

    std::vector<double> x(N, 0.0);              // full vector (replicated)
    std::vector<double> Ax_local(local_rows);   // y_k, local partial product
    std::vector<double> Ax(N);                  // y_k, global product (replicated)

    // Reproducible random initial vector
    if (rank == 0) {
		std::mt19937 gen(111);         // deterministic seed for reproducibility
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (double &val : x) val = dist(gen);
    }
	// send the initial vector to all processes
    MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // pre-compute counts and displacements for MPI_Allgatherv
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_per_proc + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }

	// Normalize initial vector so that ||x_0|| = 1
	double global_n2 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double inv_norm = 1.0 / std::sqrt(global_n2);
    for (double &val : x) val *= inv_norm;
	
    // power iteration loop
    double lambda = 0.0;
    for (int iter = 0; iter < max_iters; ++iter) {
        // local matrix-vector product: Ax_local = A_local * x
        for (int i = 0; i < local_rows; ++i) {			
            int global_i = row_start + i;
            double sum   = 0.0;
            for (int j = 0; j < N; ++j)
                sum += Aij(global_i, j) * x[j];
            Ax_local[i] = sum;
        }

        // gather the distributed pieces into the full Av on every rank
        MPI_Allgatherv(Ax_local.data(), local_rows, MPI_DOUBLE,
                       Ax.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Rayleigh quotient lambda = x * Ax / (x*x) (NOTE: x*x is 1)
        double num = 0.0;
        for (int i = 0; i < N; ++i) {
            num += x[i] * Ax[i];
        }
        lambda = num;

        // Normalise Ax
        double norm2 = 0.0;
        for (double val : Ax) norm2 += val * val;
        const double norm = std::sqrt(norm2);
        for (double &val : Ax) val /= norm;

        // Convergence: ||Ax − x||
        double diff2 = 0.0;
        for (int i = 0; i < N; ++i) {
            double diff = Ax[i] - x[i];
            diff2 += diff * diff;
        }
        const double residual = std::sqrt(diff2);

        if (rank == 0 && (iter % 20 == 0 || residual < tollerance)) {
            std::cout << "Iter " << std::setw(4) << iter
                      << " | lambda ~= " << std::setprecision(12) << lambda
                      << " | residual=" << residual << std::endl;
        }

        if (residual < tollerance) break;     // convergence check
		std::swap(x, Ax);                     // next iteration
    }

    // -------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "----------------------------------------\n"
                  << "Dominant eigenvalue ~= " << std::setprecision(12) << lambda
                  << "\n";
    }
    MPI_Finalize();
    return 0;
}
