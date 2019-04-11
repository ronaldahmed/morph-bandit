#!/bin/bash

# import treebanks from UD v2.3 and reformat to UM

. iso_ud_mapper.sh
langs="de" # mt not to be imported for task 2

basedir="$(pwd)"
bdir_t2=$basedir/task2

cd $HOME/ud-compatibility/UD_UM

for lid in $langs; do
	echo "lid: $lid"
	phdirs="${iso_ud[$lid]}"
	
	for ud_dir in $phdirs; do
		rm -r $bdir_t2/$ud_dir
		cp -r $HOME/ud-treebanks-v2.3/$ud_dir $bdir_t2/
		flist="$(ls $bdir_t2/$ud_dir/*.conllu)"
		for fname in $flist; do
			cat $fname | grep -vP "^[0-9]+-[0-9]+" > tmp
			mv tmp $fname
		done

		for fname in $flist; do
			echo "$fname"
			python3 marry.py convert -l $lid --ud $fname
			rm $fname
			# converter now creates file with -um- infix
			#new_fname=$(echo "$fname" | sed -e "s/-ud-/-um-/")
			#echo "$new_fname"
			#mv $fname $new_fname
			echo "---------------------------------------------------"
		done
	done
done
