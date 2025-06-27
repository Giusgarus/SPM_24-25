#!/usr/bin/env bash

SIZES=("10M" "100M")
PAYLOADS=("8" "64" "256")
THREADS_LIST=("1" "8" "16" "32")
NODES_LIST=(1 4 8)
TASKS_PER_NODE_LIST=(1 16 32)

EXEC=./par_merge_psrs
EXEC_PLAIN=./plain_sort



# Run plain sort with different payloads and sizes
for PAYLOAD in "${PAYLOADS[@]}"; do
  srun make cleanall
  srun make RPAYLOAD="$PAYLOAD"
  for SIZE in "${SIZES[@]}"; do
    echo "=== SIZE=$SIZE | PAYLOAD=$PAYLOAD ===" >> ./Results/plain_results_$PAYLOAD.txt
    srun --nodes=1 \
        --ntasks-per-node=1 \
        --time=00:10:00 \
        --cpu-bind=none \
        $EXEC_PLAIN -s "$SIZE" -r "$PAYLOAD"  >> ./Results/plain_results_$PAYLOAD.txt 2>&1
    echo >> ./Results/plain_results_$PAYLOAD.txt
    rm -f ./Dataset/sorted_*
  done
done


srun make clean-par_merge_psrs
srun make psrs
for PAYLOAD in "${PAYLOADS[@]}"; do
  for NODES in "${NODES_LIST[@]}"; do
    for SIZE in "${SIZES[@]}"; do
      for THREADS in "${THREADS_LIST[@]}"; do
        for TASKS in "${TASKS_PER_NODE_LIST[@]}"; do
          echo "=== SIZE=$SIZE | PAYLOAD=$PAYLOAD | THREADS=$THREADS | NODES=$NODES | TASKS=$TASKS ===" >> ./Results/psrs_results_$PAYLOAD.txt
          srun --nodes=$NODES \
              --ntasks-per-node=$TASKS \
              --time=00:10:00 \
              --mpi=pmix \
              --cpu-bind=none \
              $EXEC -s "$SIZE" -r "$PAYLOAD" -t "$THREADS" >> ./Results/psrs_results_$PAYLOAD.txt 2>&1
          echo >> ./Results/psrs_results_$PAYLOAD.txt
          rm -f ./Dataset/sorted_*
        done
      done
    done
  done
done