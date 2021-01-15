#!/bin/bash

direc='./motifs'

for motif_num in {1..32}
do
    tomtom -evalue -thresh 0.1 -o "./results/filter-${motif_num}" "$direc/model-test/filter-${motif_num}.txt" "$direc/motif_database.meme"
done