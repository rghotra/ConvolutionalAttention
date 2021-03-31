#!/bin/bash

motif_dir=$1
output_dir=$2
database=./motif_database.txt

echo $motif_dir
echo $output_dir

tomtom -evalue -thresh 0.1 -o $output_dir $motif_dir $database 
