#!/bin/bash

batch=$1

basedir="models-ens"
mkdir -p $basedir
# got first 10,000 prime numbers, shuffled them, and extracted the first 10
seeds="5167 4783 2311 9173 4793 1019 8431 673 2531 2953"

for tbname in $(cut -f 2 -d " " $batch); do
	echo $tbname
	mkdir -p $basedir/$tbname
	for seed in $seeds; do
		qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
		-o $basedir/$tbname/lem-$seed-log.out \
		-e $basedir/$tbname/lem-$seed-log.err \
		wraps/run_lemmatizer.sh $tbname train_simple $basedir/$tbname $seed
	done
done