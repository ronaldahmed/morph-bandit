#!/bin/bash

batch=$1
exp=$2

tagger_mode="bundle"
beam_size="-1"

if [ ${exp: -2} == "a1" ]; then
	tagger_mode="bundle"
elif [ ${exp: -2} == "a2" ]; then
	tagger_mode="fine-seq"
fi

mkdir -p models-anlz


for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	mkdir -p models-anlz/$tb
	lang_name=${tb:0:2}

	op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)
	input_model=models-segm/$tb/segm_$op_ep.pth
	emb_file=""

	if [ ${exp:0:2} == "l1" ]; then
		emb_file=models-segm/$tb/emb.pth
	elif [ ${exp:0:3} == "ml1" ]; then
		if [ $lang_name == "es" ]; then
			emb_file=l1-multi-emb/cs-es/cs-es/vectors-es.pth
		else
			emb_file=l1-multi-emb/"$lang_name"-es/"$lang_name"-es/vectors-"$lang_name".pth
		fi
	fi

	qsub -q 'gpu-troja.q' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-anlz/$tb/log-"$exp".out \
	-e models-anlz/$tb/log-"$exp".err \
	wraps/run_analizer.sh $tb train $input_model $emb_file $beam_size 42 $tagger_mode $exp
	
	# bash wraps/run_analizer.sh $tb train $input_model $emb_file $beam_size 42 $tagger_mode $exp_id

done