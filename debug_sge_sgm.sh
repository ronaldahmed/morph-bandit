#!/bin/bash

batch=$1
mode=$2
for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb ----------------------------"
	cat models-segm/$tb/log-$mode.err
	echo ""
	echo ""
done
