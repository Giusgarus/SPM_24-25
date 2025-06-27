#!/usr/bin/env bash
# run_benchmarks.sh
# Compila e misura miniz_parallel su node08 con 3 run per ciascun numero di thread

set -euo pipefail

RESULT_DIR="results2"
OUTPUT_FILE="$RESULT_DIR/compression_256KB.txt"

THREADS=(1 4 8 16 32)
RUNS=3            # ripetizioni per varianza

mkdir -p "$RESULT_DIR"
touch "$OUTPUT_FILE"                       # crea il file se non esiste, non lo azzera

echo "=== COMPILAZIONE ===" | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 make clean       2>&1 | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 make miniz_parallel -j          2>&1 | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 rm -f bigfile1.dat smallfile1.dat bigfile1.dat.zip smallfile1.dat.zip
srun --nodelist=node08 rm -rf folder1

for T in "${THREADS[@]}"; do
    for RUN in $(seq 1 "$RUNS"); do
        echo -e "\n=== RUN #$RUN con OMP_NUM_THREADS=$T ===" | tee -a "$OUTPUT_FILE"

        srun --nodelist=node08 bash -c "set -e
            # ---------- Preparazione input ----------
            dd if=/dev/urandom of=bigfile1.dat bs=2M count=70 status=none
            dd if=/dev/urandom of=smallfile1.dat bs=2M count=30 status=none

            mkdir -p folder1/folder2
            dd if=/dev/urandom of=folder1/bigfile2.dat bs=2M count=80 status=none
            dd if=/dev/urandom of=folder1/smallfile2.dat bs=2M count=50 status=none
            dd if=/dev/urandom of=folder1/folder2/bigfile3.dat bs=2M count=150 status=none
            dd if=/dev/urandom of=folder1/folder2/smallfile3.dat bs=2M count=30 status=none

            # ---------- Esecuzione ----------
            OMP_NUM_THREADS=$T ./miniz_parallel -r 1 -C 0 folder1 bigfile1.dat smallfile1.dat 2>&1 | tee -a "$OUTPUT_FILE"

            STATUS=\$?

            # ---------- Pulizia ----------
            rm -f bigfile1.dat smallfile1.dat bigfile1.dat.zip smallfile1.dat.zip
            rm -rf folder1

            exit \$STATUS"
    done
done

echo -e '\nTutti i test terminati. Log in '"$OUTPUT_FILE"