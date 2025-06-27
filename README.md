# SPM_24-25 - Structured Parallel and Multicore Programming

Repository for the Structured Parallel and Multicore Programming (SPM) course - Academic Year 2024-2025.

## 📁 Repository Structure

### 📝 Assignments

The repository contains 4 main assignments, each with their own implementations and benchmarks:

#### 🎯 Assignment 1: Softmax Implementation
**Folder:** `assignment1/`

Implementation and optimization of the Softmax function with different parallelization techniques:

- **`softmax_plain.cpp`** - Basic sequential implementation
- **`softmax_auto.cpp`** - Version with compiler auto-vectorization
- **`softmax_avx.cpp`** - Optimized implementation with AVX instructions
- **`softmax2_auto.cpp`** - Alternative variant with auto-vectorization
- **`softmax2_avx.cpp`** - Alternative variant with AVX

**Features:**
- Numerical stabilization to avoid overflow
- Benchmarking with different input sizes (10, 100, 1000, 10000)
- Performance comparison between different versions
- Automated testing scripts (`run.sh`, `run_diff.sh`)

#### 🔢 Assignment 2: Collatz Conjecture
**Folder:** `assignment2/`

Parallelization of Collatz conjecture computation:

- **`collatz_plain.cpp`** - Sequential implementation
- **`collatz_par.cpp`** - Parallelized version with thread pool

**Features:**
- Multiple number range handling
- Static and dynamic scheduling
- Custom thread pool (`include/threadPool.hpp`)
- Performance benchmarking

#### 🗜️ Assignment 3: Parallel Compression (MinZip)
**Folder:** `assignment3/`

Parallel file compression system based on MinZip:

- **`miniz_plain.cpp`** - Sequential compression
- **`miniz_parallel.cpp`** - Parallelized version
- **`miniz_parallel_task`** - Task-based variant

**Features:**
- File and directory compression/decompression
- Pipeline-based parallelization
- Automated benchmarking (`run_benchmarks.sh`)
- Integrated MinZip library (`miniz/`)

#### 📊 Assignment 4: Parallel Sorting
**Folder:** `assignment4/`

Implementation of parallel sorting algorithms:

- **`plain_sort.cpp`** - Sequential reference sorting
- **`par_merge.cpp`** - Parallel merge sort
- **`par_merge_psrs.cpp`** - Parallel Sample Sort (PSRS)
- **`gen_dataset.cpp`** - Dataset generator for testing

**Features:**
- Large dataset handling
- Scalable sorting algorithms
- Performance benchmarking
- Automatic test dataset generation

### 🛠️ Code Examples
**Folder:** `Code/`

Collection of code examples provided by Professor Massimo Torquati for various course topics:

- **`auto-vect-simple/`** - Auto-vectorization examples
- **`spmcode1/` - `spmcode12/`** - Progressive parallel programming examples
- **`spmcuda/`** - CUDA programming examples



## 🔧 Compilation and Execution

### Requirements
- C++17 compatible compiler (g++, clang++)
- AVX instruction support (for assignment 1)
- pthread library
- Make

### Build
Each assignment has its own Makefile:

```bash
cd assignment1/
make all          # Compile all targets
make clean        # Clean object files
make cleanall     # Clean everything
```

### Execution
Execution examples for each assignment:

```bash
# Assignment 1 - Softmax
./softmax_plain 1000
./softmax_avx 1000

# Assignment 2 - Collatz
./collatz_plain 1-1000000
./collatz_par 1-1000000

# Assignment 3 - Compression
./miniz_plain -c input.txt output.zip
./miniz_parallel -c input_directory/ output.zip

# Assignment 4 - Sorting
./plain_sort -s 100M -r 256
./par_merge -s 100M -r 256 -t 4
```

## 📈 Benchmarking

Each assignment includes automated benchmarking scripts:

- **`run.sh`** - Main execution script
- **`results/`** - Directory with benchmark results
- **`difference/`** - Comparisons between different implementations

## 🏗️ Technical Structure

### Common Header Files
- **`hpc_helpers.hpp`** - Utilities for timing and performance measurement
- **`threadPool.hpp`** - Custom thread pool implementation
- **`cmdline.hpp`** - Command line argument parsing
- **`config.hpp`** - Global configurations

### Optimizations
- **Auto-vectorization** - Automatic exploitation of SIMD instructions
- **AVX Instructions** - Manual optimizations with vector instructions
- **Thread Parallelism** - Parallelization with native threads
- **Task-based Parallelism** - Task-based approach

## 🎓 Learning Objectives

This repository demonstrates:

1. **Optimization Techniques:**
   - Automatic and manual vectorization
   - Thread-based parallelization
   - Pipeline parallelism

2. **Performance Analysis:**
   - Systematic benchmarking
   - Comparison between different implementations
   - Scalability and overhead analysis

3. **Parallel Programming:**
   - Thread pool management
   - Load balancing
   - Synchronization patterns

## 📊 Results

Benchmark results are available in the `results/` folders of each assignment, with detailed metrics on:

- Execution time
- Achieved speedup
- Parallelization efficiency
- Resource utilization

## 📋 Assignment Details

### Assignment 1: Optimized Softmax Implementation
**Objective:** Compare different optimization strategies for the Softmax function

**Implementations:**
1. **Plain** - Basic sequential implementation
2. **Auto-vectorization** - Using automatic compiler optimizations
3. **AVX** - Manual optimization with SIMD AVX instructions
4. **Alternative variants** - Different implementations for performance comparison

**Testing Methodology:**
- Input sizes: 10, 100, 1000, 10000 elements
- Accurate execution time measurement
- Numerical difference analysis between implementations
- Automatic comparative report generation

### Assignment 2: Collatz Conjecture Parallelization
**Objective:** Implement efficient parallelization for Collatz conjecture computation

**Technical Features:**
- Custom thread pool for optimal thread management
- Support for multiple input ranges
- Both static and dynamic load balancing implementation
- Algorithmic optimizations (combining sequence steps)

### Assignment 3: Parallel MinZip Compression
**Objective:** Develop parallel file compression system

**Functionality:**
- Compression/decompression of single files and directories
- Parallel pipeline to maximize throughput
- Support for variable chunk sizes
- Task-based parallelism with dynamic load management

### Assignment 4: Parallel Sorting Algorithms
**Objective:** Implement and compare scalable parallel sorting algorithms

**Implemented Algorithms:**
1. **Sequential Sort** - Baseline with std::sort
2. **Parallel Merge Sort** - Parallel divide-and-conquer
3. **Parallel Sample Sort (PSRS)** - Scalable distributed algorithm

**Testing:**
- Datasets up to 100M+ records
- Configurable record sizes (64-256 bytes)
- Automatic test dataset generation
- Performance analysis on different workloads

## 🔬 Benchmarking Methodology

### Testing Environment
- **Hardware:** Cluster with dedicated nodes (node01, node08, etc.)
- **Scheduler:** SLURM for controlled execution
- **Compiler:** GCC with `-O3 -march=native` optimizations
- **Metrics:** Execution time, speedup, efficiency, resource utilization

### Measurement Procedures
1. **Warm-up** - Preliminary executions to stabilize performance
2. **Multiple repetitions** - Average over 3+ executions to reduce variance
3. **Load control** - Execution on dedicated nodes for accurate measurements
4. **Result validation** - Output comparison for algorithmic correctness

## 🛠️ Setup Guides

### Environment Preparation
```bash
# Clone repository
git clone [repository-url]
cd SPM_24-25

# Verify dependencies
g++ --version    # Requires GCC 7+ for C++17
make --version
```

### Complete Assignment Execution
```bash
# Assignment 1 - Complete softmax test
cd assignment1/
chmod +x run.sh
./run.sh

# Assignment 2 - Collatz benchmark
cd ../assignment2/
chmod +x run.sh
./run.sh

# Assignment 3 - Compression benchmark
cd ../assignment3/
chmod +x run_benchmarks.sh
./run_benchmarks.sh

# Assignment 4 - Sorting test
cd ../assignment4/
chmod +x run.sh
./run.sh
```

### Parameter Customization
```bash
# Modify test sizes
export TEST_SIZES="100 1000 10000 100000"

# Configure thread count
export MAX_THREADS=16

# Set payload size (Assignment 4)
make RPAYLOAD=128
```

## 📊 Results Interpretation

### Key Metrics
- **Speedup:** Sequential_time / Parallel_time ratio
- **Efficiency:** Speedup / number_of_threads
- **Scalability:** Performance growth with increasing resources
- **Overhead:** Additional cost of parallelization

### Output Files
- `results/*.txt` - Raw execution data
- `difference/*.txt` - Numerical comparisons between implementations
- `*.log` - Detailed execution logs

## 🔍 Performance Analysis

### Assignment 1 - Softmax
- **Vectorization Impact:** SIMD vs scalar performance comparison
- **Auto vs Manual:** Effectiveness of automatic vs manual optimizations
- **Numerical Stability:** Stabilization impact on performance

### Assignment 2 - Collatz
- **Load Balancing:** Static vs dynamic scheduling comparison
- **Thread Scaling:** Scalability analysis with varying thread count
- **Work Distribution:** Load distribution efficiency

### Assignment 3 - Compression
- **Pipeline Efficiency:** Pipeline throughput vs overhead
- **Chunk Size Impact:** Chunk size optimization
- **I/O Parallelization:** I/O vs computation balancing

### Assignment 4 - Sorting
- **Algorithm Comparison:** PSRS vs Parallel Merge Sort
- **Memory Access Patterns:** Cache efficiency of different algorithms
- **Dataset Size Scaling:** Behavior on growing datasets

## 🎯 Skills Developed

### Parallel Programming
- **Thread Management:** Pools, scheduling, synchronization
- **SIMD Programming:** Manual and automatic vectorization
- **Pipeline Design:** Multi-stage parallelization
- **Load Balancing:** Optimal work distribution

### Performance Optimization
- **Profiling:** Performance bottleneck identification
- **Memory Optimization:** Cache-friendly data structures
- **Compiler Optimization:** Compiler flag exploitation
- **Hardware Awareness:** Architectural feature utilization

### Experimental Analysis
- **Benchmarking Methodology:** Valid experiment design
- **Statistical Analysis:** Variance and outlier management
- **Performance Modeling:** Algorithm behavior prediction
- **Scalability Assessment:** Efficiency evaluation on different resources

## 🤝 Contributors

Repository developed for the SPM course - University of Pisa, Academic Year 2024-2025.

**Author:** Giuseppe Gabriele Russo  
**Course:** Structured Parallel and Multicore Programming  
**University:** University of Pisa  
**Academic Year:** 2024-2025

## 📚 References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) - SIMD instruction documentation
- [OpenMP Specification](https://www.openmp.org/specifications/) - Parallelization standard
- [C++ Concurrency](https://en.cppreference.com/w/cpp/thread) - C++17 threading
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html) - Job scheduler

