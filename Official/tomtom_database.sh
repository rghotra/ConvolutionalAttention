#!/bin/bash

direc="./motifs"

if [ -d "./results" ]
then
    rm -r "./results"
fi

for param in ${direc}/*/
do
    for model in ${param}*
    do
	direcs=(${model//// })
	name=${direcs[-1]}
	name=(${name//./ })
	name=${name[0]}
  	category=${direcs[-2]}

	out=./results/${category}/${name}

	mkdir -p -- "${out}"
    	tomtom -evalue -thresh 0.1 -o "${out}" "${model}" "${direc}/motif_database.meme"
    done
done