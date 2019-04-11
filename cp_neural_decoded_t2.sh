#!/bin/bash

. iso_ud_mapper.sh
. uddir_isoud_mapper.sh

source_dir=$1 # folder with decoded files

langs="en cs es tr ar ja"


basedir="$(pwd)"
bdir_t2=$basedir/task2

for lid in $langs; do
	echo "lid: $lid"
	phdirs="${iso_ud[$lid]}"
	
	for ud_dir in $phdirs; do
		isotb="${uddir_isotb[$ud_dir]}"
		# fix notation on lemm baseline
		cp $bdir_t2/$ud_dir/pred-dev.conllu $bdir_t2/$ud_dir/pred-lemming-dev.conllu
		cp $source_dir/"$isotb"-um-dev.conllu.output $bdir_t2/$ud_dir/pred-neural-dev.conllu

		echo $source_dir/"$isotb"-um-dev.conllu.output
		echo $bdir_t2/$ud_dir/pred-neural-dev.conllu
	done
done