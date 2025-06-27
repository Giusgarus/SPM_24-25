#!/bin/bash

# Lista dei valori
values=(10 100 1000 10000)

srun --nodelist=node01 make cleanall
srun --nodelist=node01 make

# Loop attraverso i valori ed esegui il comando
for val in "${values[@]}"; do
    echo "Eseguendo softmax con valore: $val"
    srun --nodelist=node01 ./softmax_plain "$val" 1 &> ./results/plain_$val.txt
    srun --nodelist=node01 ./softmax_auto "$val" 1 &> ./results/auto_$val.txt
    srun --nodelist=node01 ./softmax2_auto "$val" 1 &> ./results/auto2_$val.txt
    srun --nodelist=node01 ./softmax_avx "$val" 1 &> ./results/avx_$val.txt
    srun --nodelist=node01 ./softmax2_avx "$val" 1 &> ./results/2avx_$val.txt
done
