#!/bin/bash

batch=$1
beam_size="-1"

mkdir -p models-anlz

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	mkdir -p models-anlz/$tb

	op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)

	input_model=models-segm/$tb/segm_$op_ep.pth
	emb_file=models-segm/$tb/emb.pth

	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=8G,mem_free=12G,act_mem_free=12G,h_data=17G -p -10 \
	-o models-anlz/$tb/log-l1a2.out \
	-e models-anlz/$tb/log-l1a2.err \
	wraps/run_analizer_l1a2.sh $tb train $input_model $emb_file $beam_size 42

done