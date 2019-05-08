#!/bin/bash

batch=$1

mkdir -p models-anlz

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	mkdir -p models-anlz/$tb

	op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)

	if [ $op_ep == "0" ]; then
		op_ep="19"
	fi
	if [ $tb == "kpv_ikdp" ]; then
		op_ep="9"
	fi

	input_model=models-segm/$tb/segm_$op_ep.pth

	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=3G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-anlz/$tb/log.out \
	-e models-anlz/$tb/log.err \
	wraps/run_analizer.sh $tb $input_model

done