#!/usr/bin/env bash
#
# Crea numbers.txt con numeri consecutivi (uno per riga)
# fino a raggiungere 150 MiB di dimensione.

set -euo pipefail

outfile="numbers.txt"          # nome del file da generare
target=$((150*1024*1024))      # 150 MiB in byte
chunk=10000                    # quanti numeri scrivere per iterazione

: > "$outfile"                 # svuota/crea il file
n=1                            # numero iniziale

echo "Sto generando $outfile fino a $target byte (~150 MiB)..."

while [ "$(stat -c%s "$outfile")" -lt "$target" ]; do
  # scrive numeri da $n a $((n+chunk-1)) nel file
  seq "$n" $((n+chunk-1)) >> "$outfile"
  n=$((n+chunk))
done

# se abbiamo superato di qualche byte, tronchiamo al valore esatto
truncate -s "$target" "$outfile"

actual_size=$(stat -c%s "$outfile")
printf 'Fatto! Dimensione finale: %s byte (%.2f MiB)\n' \
       "$actual_size" "$(echo "scale=2; $actual_size/1048576" | bc -l)"