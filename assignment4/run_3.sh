#!/usr/bin/env bash

SIZES=("20M" "80M" "160M")
PAYLOADS=("64")
THREADS_LIST=("8")
NODES_LIST=(1 4 8)

EXEC=./par_merge
EXEC_PSRS=./par_merge_psrs

# --- par_merge ---
for PAYLOAD in "${PAYLOADS[@]}"; do
  srun make cleanall
  srun make RPAYLOAD="$PAYLOAD"
  for INDEX in "${!NODES_LIST[@]}"; do
    NODES=${NODES_LIST[$INDEX]}
    SIZE=${SIZES[$INDEX]}
    for THREADS in "${THREADS_LIST[@]}"; do
        echo "=== SIZE=$SIZE | PAYLOAD=$PAYLOAD | THREADS=$THREADS | NODES=$NODES | TASKS=1 ===" >> ./Results/weak_scale.txt
        srun --nodes=$NODES \
            --ntasks-per-node=1 \
            --time=00:10:00 \
            --mpi=pmix \
            --cpu-bind=none \
            $EXEC -s "$SIZE" -r "$PAYLOAD" -t "$THREADS" >> ./Results/weak_scale.txt 2>&1
        echo >> ./Results/weak_scale.txt
        rm -f ./Dataset/sorted_*
      done
    done
done

# --- par_merge_psrs ---
srun make clean-par_merge_psrs
srun make psrs

for PAYLOAD in "${PAYLOADS[@]}"; do
  for INDEX in "${!NODES_LIST[@]}"; do
    NODES=${NODES_LIST[$INDEX]}
    SIZE=${SIZES[$INDEX]}
    for THREADS in "${THREADS_LIST[@]}"; do
        echo "=== SIZE=$SIZE | PAYLOAD=$PAYLOAD | THREADS=$THREADS | NODES=$NODES | TASKS=1 ===" >> ./Results/weak_scale_psrs.txt
        srun --nodes=$NODES \
            --ntasks-per-node=1 \
            --time=00:10:00 \
            --mpi=pmix \
            --cpu-bind=none \
            $EXEC_PSRS -s "$SIZE" -r "$PAYLOAD" -t "$THREADS" >> ./Results/weak_scale_psrs.txt 2>&1
        echo >> ./Results/weak_scale_psrs.txt
        rm -f ./Dataset/sorted_*
      done
    done
done