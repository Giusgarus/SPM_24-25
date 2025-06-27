#!/bin/bash

# Lista dei valori
values=(10 100 1000 10000)  # Sostituisci con i tuoi valori reali

# Loop attraverso i valori
for val in "${values[@]}"; do
    echo "Eseguendo diff per il valore: $val"

    # Esegui il diff fra plain e auto
    echo "Differenza tra plain e auto:"
    diff "./results/plain_$val.txt" "./results/auto_$val.txt" > "./difference/diff_plain_auto_$val.txt"

    # Esegui il diff fra plain e auto2
    echo "Differenza tra plain e auto2:"
    diff "./results/plain_$val.txt" "./results/auto2_$val.txt" > "./difference/diff_plain_auto2_$val.txt"
    
    # Esegui il diff fra plain e avx
    echo "Differenza tra plain e avx:"
    diff "./results/plain_$val.txt" "./results/avx_$val.txt" > "./difference/diff_plain_avx_$val.txt"
    
    # Esegui il diff fra plain e 2avx
    echo "Differenza tra plain e 2avx:"
    diff "./results/plain_$val.txt" "./results/2avx_$val.txt" > "./difference/diff_plain_2avx_$val.txt"
done