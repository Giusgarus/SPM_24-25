#!/bin/bash

# Check if the first parameter is "-d"
dynamic_flag=""
if [ "$1" == "-d" ]; then
    dynamic_flag="-d"
    shift  # Remove the first parameter (-d) from the arguments
fi

# Check if the minimum parameters have been passed
if [ $# -gt 1 ]; then
    echo "Usage: $0 [-d]"
    exit 1
fi

# Define the ranges
ranges=("1-1000" "50000000-100000000" "1000000000-1100000000")

# Define values for num_threads and chunk_size
num_threads_values=(1 4 16 32)  # Adjust as needed
chunk_size_values=(131072)  # Adjust as needed

# Cleanup and compilation
srun --nodelist=node08 make cleanall
srun --nodelist=node08 make

# Build the range string for the command
range_args=""
for range in "${ranges[@]}"; do
    range_args+=" $range"
done

echo "Running PLAIN with ranges: ${ranges[*]}"
srun --nodelist=node08 ./collatz_plain $range_args

# Iterate over num_threads and chunk_size values
for num_threads in "${num_threads_values[@]}"; do
    for chunk_size in "${chunk_size_values[@]}"; do
        echo "Running PAR with ranges: ${ranges[*]}, num_threads: $num_threads, chunk_size: $chunk_size, dynamic: ${dynamic_flag:-no}"
        output_file="./results/result_static.txt"
        if [ -n "$dynamic_flag" ]; then
            output_file="./results/result_dynamic.txt"
        fi

        srun --nodelist=node08 ./collatz_par $dynamic_flag -n "$num_threads" -c "$chunk_size" $range_args >> "$output_file"
    done
done