#!/bin/bash

batch=$1
beam_size="-1"
exp_id="multi-l1-a1"
tagger_mode="bundle"

mkdir -p models-anlz

eval_mode="fine"

if [ $exp_id == "l1-a2" ]; then
	tagger_mode="fine-seq"
fi


for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	mkdir -p models-anlz/$tb
	lang_name=${tb:0:2}

	op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)
	input_model=models-segm/$tb/segm_$op_ep.pth
	emb_file=""
	if [ $exp_id == "l1-a2" ]; then
		emb_file=models-segm/$tb/emb.pth
	elif [ $exp_id == "multi-l1-a1" ]; then
		emb_file=l1-multi-emb/"$lang_name"-es/"$lang_name"-es/vectors-"$lang_name".pth
	fi

	# qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	# -o models-anlz/$tb/log.out \
	# -e models-anlz/$tb/log.err \
	# wraps/run_analizer.sh $tb train $input_model $emb_file $beam_size 42 $tagger_mode $eval_mode $exp_id
	bash wraps/run_analizer.sh $tb train $input_model $emb_file $beam_size 42 $tagger_mode $eval_mode $exp_id

done