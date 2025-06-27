#!/usr/bin/env bash
# run_once_16threads.sh
# Compila miniz_parallel ed esegue un singolo test (16 thread) su node08

set -euo pipefail

RESULT_DIR="results2"
OUTPUT_FILE="$RESULT_DIR/test.txt"
T=16            # numero di thread fisso

mkdir -p "$RESULT_DIR"
touch "$OUTPUT_FILE"             # crea il log se manca, non lo azzera

echo "=== COMPILAZIONE ===" | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 make clean            2>&1 | tee -a "$OUTPUT_FILE"
srun --nodelist=node08 make miniz_parallel -j 2>&1 | tee -a "$OUTPUT_FILE"

# rimozione preventiva di eventuali residui
srun --nodelist=node08 bash -c 'rm -f numbers.txt bigfile1.dat smallfile1.dat *.zip; rm -rf folder1'

echo -e "\n=== RUN unico con OMP_NUM_THREADS=$T ===" | tee -a "$OUTPUT_FILE"

srun --nodelist=node08 bash -c "set -e
    # ---------- Preparazione input ----------
    dd if=/dev/urandom of=bigfile1.dat bs=2M count=70  status=none
    dd if=/dev/urandom of=smallfile1.dat bs=2M count=30 status=none

    mkdir -p folder1/folder2
    dd if=/dev/urandom of=folder1/bigfile2.dat bs=2M count=80  status=none
    dd if=/dev/urandom of=folder1/smallfile2.dat bs=2M count=50 status=none
    dd if=/dev/urandom of=folder1/folder2/bigfile3.dat bs=2M count=150 status=none
    dd if=/dev/urandom of=folder1/folder2/smallfile3.dat bs=2M count=30  status=none

    # ---------- MD5 degli input ----------
    echo '--- MD5SUM ---'
    md5sum bigfile1.dat smallfile1.dat 2>&1 | tee -a "$OUTPUT_FILE"

    # ---------- COMPRESSIONE ----------
    echo '--- COMPRESSIONE ---'
    OMP_NUM_THREADS=$T ./miniz_parallel -r 1 -C 1 folder1 bigfile1.dat smallfile1.dat 2>&1 | tee -a "$OUTPUT_FILE"

    # ---------- DECOMPRESSIONE ----------
    echo '--- DECOMPRESSIONE ---'
    OMP_NUM_THREADS=$T ./miniz_parallel -r 1 -D 1 folder1 bigfile1.dat.zip smallfile1.dat.zip 2>&1 | tee -a "$OUTPUT_FILE"

    # ---------- MD5 degli input ----------
    echo '--- MD5SUM ---'
    md5sum bigfile1.dat smallfile1.dat 2>&1 | tee -a "$OUTPUT_FILE"

    STATUS=\$?

    # ---------- Pulizia ----------
    rm -f bigfile1.dat smallfile1.dat *.zip
    rm -rf folder1

    exit \$STATUS
" 2>&1 | tee -a "$OUTPUT_FILE"

echo -e "\nRun completata. Log salvato in $OUTPUT_FILE"