#!/bin/bash

batch=$1

basedir="models-ens"
beam_size="-1"
mkdir -p $basedir
# got first 10,000 prime numbers, shuffled them, and extracted the first 10
seeds="5167 4783 2311 9173 4793 1019 8431 673 2531 2953"

for tbname in $(cut -f 2 -d " " $batch); do
	echo $tbname
	mkdir -p $basedir/$tbname

	op_ep=$(tail -1 models-segm/$tbname/log.out | cut -f 1)
	input_model=models-segm/$tbname/segm_$op_ep.pth
	emb_file=models-segm/$tbname/emb.pth

	for seed in $seeds; do
		qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=3G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
		-o $basedir/$tbname/anlz-$seed-log.out \
		-e $basedir/$tbname/anlz-$seed-log.err \
		wraps/run_analizer.sh $tbname train_simple $input_model $emb_file $beam_size $seed
	done
done