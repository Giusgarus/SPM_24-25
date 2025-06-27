#!/usr/bin/env bash

SIZES=("10M" "100M")
PAYLOADS=("8" "64")
THREADS_LIST=("1" "8" "16" "32")
NODES_LIST=(1 4 8)
TASKS_PER_NODE_LIST=(1 16 32)

EXEC=./par_merge

for PAYLOAD in "${PAYLOADS[@]}"; do
  srun make cleanall
  srun make RPAYLOAD="$PAYLOAD"
  for NODES in "${NODES_LIST[@]}"; do
    for SIZE in "${SIZES[@]}"; do
      for THREADS in "${THREADS_LIST[@]}"; do
        for TASKS in "${TASKS_PER_NODE_LIST[@]}"; do
          echo "=== SIZE=$SIZE | PAYLOAD=$PAYLOAD | THREADS=$THREADS | NODES=$NODES | TASKS=$TASKS ===" >> ./Results/results_$PAYLOAD.txt
          srun --nodes=$NODES \
              --ntasks-per-node=$TASKS \
              --time=00:10:00 \
              --mpi=pmix \
              --cpu-bind=none \
              $EXEC -s "$SIZE" -r "$PAYLOAD" -t "$THREADS" >> ./Results/results_$PAYLOAD.txt 2>&1
          echo >> ./Results/results_$PAYLOAD.txt
          rm -f ./Dataset/sorted_*
        done
      done
    done
  done
done
