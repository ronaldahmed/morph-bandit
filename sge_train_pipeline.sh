#!/bin/bash

batch=$1

mkdir -p models-segm
mkdir -p models-anlz

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	mkdir -p models-segm/$tb
	mkdir -p models-anlz/$tb

	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-segm/$tb/log.out \
	-e models-segm/$tb/log.err \
	wraps/run_lemmatizer.sh $tb models-segm/$tb

	python3 dump_segm_emb.py $tb

	op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)

	input_model=models-segm/$tb/segm_$op_ep.pth
	emb_file=models-segm/$tb/emb.pth

	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=3G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-anlz/$tb/log.out \
	-e models-anlz/$tb/log.err \
	wraps/run_analizer.sh $tb $input_model $emb_file

done