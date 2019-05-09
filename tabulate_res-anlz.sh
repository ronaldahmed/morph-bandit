#!/bin/bash

mode=$1  # dev, test, covered-test

outfile="res-anlz.$mode.csv"
echo "treebank,lem_acc,edist,msd_acc,msd_f1" > $outfile

for tb in $(cut -f 2 -d " " data/uddir_tbname); do
	echo $tb
	
	tail -1 models-anlz/$tb/log-$mode.out | sed -r "s/.*lem[_]acc: ([.0-9]+), dist: ([.0-9]+), msd[_]acc: ([.0-9]+), msd[_]f1: ([.0-9]+).*/$tb,\1,\2,\3,\4/" >> $outfile

done
