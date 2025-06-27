// Power‑iteration where *both* x and Ax are **partitioned** (block‑row).  Each
// rank owns only its rows of the matrix and the corresponding slices of the
// vectors.  Collective communication is required each iteration to assemble a
// temporary full copy of x for the local SpMV as well as to obtain global dot
// products and norms.
//
#include <mpi.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <numeric>

inline double Aij(int i, int j) {
    return 1.0 / (1.0 + std::abs(i - j));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------------------- command‑line -------------------------------------
    const int    n          = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const int    max_iters  = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const double tollerance = (argc > 3) ? std::atof(argv[3]) : 1e-8;

    // ---------------------- row distribution --------------------------------
    const int rows_per_proc = n / size;
    const int remainder     = n % size;
    const int local_rows    = rows_per_proc + (rank < remainder ? 1 : 0);
    const int row_start     = rank * rows_per_proc + std::min(rank, remainder);

    // Scatter/Allgather metadata once
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = rows_per_proc + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }

    // ---------------------- local data --------------------------------------
    std::vector<double> x_local(local_rows);     // x_k (partitioned)
    std::vector<double> Ax_local(local_rows);    // y_k (partitioned)
    std::vector<double> x_full(n);               // temporary workspace for the full vector x

    // initialize vector x
    if (rank == 0) {
        std::mt19937 gen(111);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> x_global(n);
        for (double &val : x_global) val = dist(gen);

        // Scatter slices to ranks
        MPI_Scatterv(x_global.data(), counts.data(), displs.data(), MPI_DOUBLE,
                     x_local.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     x_local.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Normalise initial vector globally so that ||x_0|| = 1
    double local_n2 = std::inner_product(x_local.begin(), x_local.end(), x_local.begin(), 0.0);
    double global_n2;
    MPI_Allreduce(&local_n2, &global_n2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double inv_norm = 1.0 / std::sqrt(global_n2);
    for (double &val : x_local) val *= inv_norm;

    if (rank == 0)
        std::cout << "Partitioned power iteration: n=" << n << ", tollerance=" << tollerance << "\n";

    // ---------------------- iteration loop ----------------------------------
    double lambda = 0.0;
    for (int iter = 0; iter < max_iters; ++iter) {
        // gather x_local slices into x_full so every rank can compute Ax_local
        MPI_Allgatherv(x_local.data(), local_rows, MPI_DOUBLE,
                       x_full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // local SpMV (row block)
        for (int i = 0; i < local_rows; ++i) {
            const int global_i = row_start + i;
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += Aij(global_i, j) * x_full[j];
            Ax_local[i] = sum;
        }

        // Rayleigh quotient lambda_k = (x_k * y_k) / (x_k * x_k) (NOTE:the denominator is 1)
        double local_num = 0.0;
        for (int i = 0; i < local_rows; ++i)
            local_num += x_local[i] * Ax_local[i];
        MPI_Allreduce(&local_num, &lambda, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Normalise Ax_local 
        double local_y2 = std::inner_product(Ax_local.begin(), Ax_local.end(), Ax_local.begin(), 0.0);
        double global_y2;
        MPI_Allreduce(&local_y2, &global_y2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double inv_y_norm = 1.0 / std::sqrt(global_y2);
        for (double &val : Ax_local) val *= inv_y_norm;

        // Convergence: ||Ax_local - x_local||
        double local_diff2 = 0.0;
        for (int i = 0; i < local_rows; ++i) {
            double diff = Ax_local[i] - x_local[i];
            local_diff2 += diff * diff;
        }
        double global_diff2;
        MPI_Allreduce(&local_diff2, &global_diff2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double resid = std::sqrt(global_diff2);

        if (rank == 0 && (iter % 20 == 0 || resid < tollerance))
            std::cout << "Iter " << std::setw(4) << iter
                      << " | lambda ~= " << std::setprecision(12) << lambda
                      << " | resid=" << resid << '\n';

        if (resid < tollerance) break;   // convergence check

        // swap for next iteration
		std::swap(x_local, Ax_local);
    }

    if (rank == 0)
        std::cout << "Dominant eigenvalue ≈ " << std::setprecision(12) << lambda << '\n';

    MPI_Finalize();
    return 0;
}
