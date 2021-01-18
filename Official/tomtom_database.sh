#!/bin/bash

model_name=$1

direc="./motifs"

mkdir -p -- "./results/${model_name}"

for motif_num in {1..32}
do
    tomtom -evalue -thresh 0.1 -o "./results/${model_name}/filter-${motif_num}" "$direc/model-test/filter-${motif_num}.txt" "$direc/motif_database.meme"
done