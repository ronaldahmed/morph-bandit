#!/bin/bash

# ISO -> UD directories mapper
. iso_ud_mapper.sh

#langs: en,es,de,cs,tr,shk,[ar,ja]
langs="en es cs tr de ar ja shk mt"
basedir="$(pwd)"

## task 2 stats
bdir_t2=$basedir/task2

for split in train dev test; do
	outfile="task2_stats-$split.csv"
	echo "folder,sents,tokens,types" > $outfile

	for lid in $langs; do
	  echo "lang: -$lid-"
	  phdirs="${iso_ud[$lid]}"
	  for ud_dir in $phdirs; do
	  	sents=($(grep -rn "# sent" $bdir_t2/$ud_dir/*$split.conllu | wc -l))
	  	tokens=($(cat $bdir_t2/$ud_dir/*$split.conllu | grep -vP "^#" | grep -vP "^$" | cut -f 2  | wc -l))
	  	types=($(cat $bdir_t2/$ud_dir/*$split.conllu | grep -vP "^#" | grep -vP "^$" | cut -f 2  | sort | uniq | wc -l))
	  	echo "$ud_dir,${sents[0]},${tokens[0]},${types[0]}" >> $outfile
	  done
	done
done