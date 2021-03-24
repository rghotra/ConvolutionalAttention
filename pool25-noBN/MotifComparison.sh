#!/bin/bash

direc=$1

if [ ! -d ./motifs/$direc ]
then
    echo "category does not exist"
    exit
fi

if [ ! -d ./results/$direc ]
then
    mkdir -p -- ./results/$direc
fi

existing=(`ls -d -- ./results/$direc/*/`)
models=(`ls ./motifs/$direc`)

for model in ${existing[@]}
do
    direcs=(${model//// })
    name=${direcs[-1]}
    name="${name}.txt"

    mods=()
    mods=${models[@]/$name}
    models=()
    models=$mods
done

for model in ${models[@]}
do
    motif=./motifs/$direc/$model

    database=./motifs/motif_database.meme

    name=(${model//./ })
    out=./results/$direc/${name[0]}

    tomtom -evalue -thresh 0.1 -o $out $motif $database
done