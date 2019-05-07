#!/bin/bash

mode=$2  # dev, test, covered-test

outfile="models-segm/res.$mode.csv"
echo "treebank,lem_acc,edist" > $outfile

for tb in $(cut -f 2 -d " " data/uddir_tbname); do
	echo $tb
	
	tail -1 models-segm/$tb/log-$mode.out | sed -r "s/.*acc: ([.0-9]+), dist: ([.0-9]+).*/$tb,\1,\2/" >> $outfile

done