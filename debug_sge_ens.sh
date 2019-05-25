#!/bin/bash

batch=$1
model=$2
seeds="5167 4783 2311 9173 4793 1019 8431 673 2531 2953"

for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb ----------------------------"
	for seed in $seeds; do
		# cat models-ens/$tb/$model-$seed-log.err
		# echo ""
		tail -1 models-ens/$tb/$model-$seed-log.out
	done
	echo ""
done
