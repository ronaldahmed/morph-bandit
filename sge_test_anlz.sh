#!/bin/bash

batch=$1
exp=$2

bsearch=-1
tagger_mode="bundle"
mode=dev  # dev, test, covered-test

if [ ${exp: -2} == "a1" ]; then
	tagger_mode="bundle"
elif [ ${exp: -2} == "a2" ]; then
	tagger_mode="fine-seq"
fi

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	lang_name=${tb:0:2}
	op_ep_seg=$(tail -1 models-segm/$tb/log.out | cut -f 1)
	input_lem_model=models-segm/$tb/segm_"$op_ep_seg".pth
	op_ep_anl=""
	input_anlz_model=""
	emb_file=""

	if [ $exp == "l1-a1" ]; then
		op_ep_anl=$(tail -1 models-anlz/$tb/log.out | cut -f 1)
		input_anlz_model=models-anlz/$tb/anlz_"$op_ep_anl".pth
	elif [ $exp == "l1-a2" ]; then
		op_ep_anl=$(tail -1 models-anlz/$tb/log-l1a2.out | cut -f 1)
		input_anlz_model=models-anlz/$tb/anlz_fine-seq_"$op_ep_anl".pth
	else
		op_ep_anl=$(tail -1 models-anlz/$tb/log-$exp.out | cut -f 1)
		input_anlz_model=models-anlz/$tb/$exp_"$op_ep_anl".pth
	fi

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
	-o models-anlz/$tb/log-$exp-$mode.out \
	-e models-anlz/$tb/log-$exp-$mode.err \
	wraps/test_anlz.sh $tb $mode "$input_lem_model" "$input_anlz_model" $emb_file $bsearch $tagger_mode $exp
	
	# bash wraps/test_anlz.sh $tb $mode "$input_lem_model" "$input_anlz_model" $emb_file
done
