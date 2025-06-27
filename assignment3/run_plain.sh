#!/usr/bin/env bash
# run_plain.sh
# Compila miniz_plain e lo esegue 3 volte su node08

set -euo pipefail

RESULT_DIR="results2"
OUTPUT_FILE="$RESULT_DIR/compression_plain.txt"
RUNS=3                           # numero di prove

mkdir -p "$RESULT_DIR"
touch "$OUTPUT_FILE"             # non azzera, crea se manca

echo "=== COMPILAZIONE miniz_plain ===" | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 make clean 2>&1 | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 make miniz_plain -j  2>&1 | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 rm -f bigfile1.dat smallfile1.dat bigfile1.dat.zip smallfile1.dat.zip
srun --nodelist=node08 rm -rf folder1

for RUN in $(seq 1 "$RUNS"); do
    echo -e "\n=== RUN #$RUN miniz_plain ===" | tee -a "$OUTPUT_FILE"

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
        ./miniz_plain -r 1 -C 0 folder1 bigfile1.dat smallfile1.dat 2>&1 | tee -a "$OUTPUT_FILE"

        STATUS=\$?

        # ---------- Pulizia ----------
        rm -f bigfile1.dat smallfile1.dat bigfile1.dat.zip smallfile1.dat.zip
        rm -rf folder1

        exit \$STATUS"
    done

echo -e "\nTutte le prove miniz_plain completate. Log in $OUTPUT_FILE"