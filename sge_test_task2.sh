#!/bin/bash

batch=$1
mode=$2  # dev, test, covered-test

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	# bash wraps/run_analizer.sh $tb models-segm/$tb
	op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)

	if [ $op_ep == "0" ]; then
		op_ep="19"
	fi
	input_model=models-segm/$tb/segm_$op_ep.pth
	
	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-segm/$tb/log-$mode.out \
	-e models-segm/$tb/log-$mode.err \
	wraps/test_lemm.sh $tb $mode $input_model

done
