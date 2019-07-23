#!/bin/bash

batch=$1
mode=$2
exp="l1.mrt.warm_optm-adadelta_alpha-0.0001_sample-20_clip-0_bs-5_temp-1"

for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb ----------------------------"
	# cat models-segm/$tb/$exp/log-$mode.err
	cat models-segm/$tb/$exp/log.err
	echo ""
	echo ""
done
