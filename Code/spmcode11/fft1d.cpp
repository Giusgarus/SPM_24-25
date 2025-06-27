//
// Distributed 1-D complex-to-complex FFT of length N = P * L
// (each rank owns a contiguous block of length L).
//
// Two-step Cooley–Tukey split.
//   1. local FFT of size L   (using FFTW)
//   2. all-to-all transpose  (column gathers)
//   3. size-P FFT (very small) on each output column
//
//
#include <mpi.h>
#include <fftw3.h>
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>

using cd = std::complex<double>;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
    const ptrdiff_t N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N % size != 0) {
        if (rank == 0) std::cerr << "N must be a multiple of P\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const ptrdiff_t L = N / size;          // local slab length
	
    std::vector<cd> x(L);
    for (ptrdiff_t i = 0; i < L; ++i)
        x[i] = std::sin(2.0*M_PI*(i + rank*L)/N);   // test wave

    //  L-point local FFT
    fftw_plan planL = fftw_plan_dft_1d(static_cast<int>(L),
									   reinterpret_cast<fftw_complex*>(x.data()),
									   reinterpret_cast<fftw_complex*>(x.data()),
									   FFTW_FORWARD, FFTW_ESTIMATE);
	
    fftw_execute(planL);         // X1(rank, j), j = 0…L-1
	
	// arrange send/recv so that element j of each rank goes to rank j%P.
    // L = N/P so j already indexes exactly one destination rank

    std::vector<cd> trans(L);              // will hold column rank data
    MPI_Alltoall(x.data(),    L/size, MPI_C_DOUBLE_COMPLEX,
                 trans.data(), L/size, MPI_C_DOUBLE_COMPLEX,
                 MPI_COMM_WORLD);
	
    // small FFT.  Now trans[k] is the value from rank k at this column (implicit)
    fftw_plan planP = fftw_plan_dft_1d(size,
									   reinterpret_cast<fftw_complex*>(trans.data()),
									   reinterpret_cast<fftw_complex*>(trans.data()),
									   FFTW_FORWARD, FFTW_ESTIMATE);
	
    for (ptrdiff_t j = 0; j < L/size; ++j)
        fftw_execute_dft(planP,
						 reinterpret_cast<fftw_complex*>(trans.data()+j*size),
						 reinterpret_cast<fftw_complex*>(trans.data()+j*size));
	
    // trans[] now holds our portion of the global FFT
	
    if (rank == 0) std::cout << "Done 1D FFT of length " << N << '\n';
	
    fftw_destroy_plan(planL);
    fftw_destroy_plan(planP);
    MPI_Finalize();
    return 0;
}
