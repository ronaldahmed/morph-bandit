#!/bin/bash

batch=$1
exp=$2

bsearch=-1
tagger_mode="bundle"
eval_mode="bundle"
mode=dev  # dev, test, covered-test

if [ ${exp:-2} == "a1" ]; then
	tagger_mode="bundle"
elif [ ${exp:-2} == "a2" ]; then
	tagger_mode="fine-seq"
fi


for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	# bash wraps/run_analizer.sh $tb models-segm/$tb
	op_ep_seg=$(tail -1 models-segm/$tb/log.out | cut -f 1)
	# op_ep_anl=$(tail -1 models-anlz/$tb/log.out | cut -f 1)
	op_ep_anl=$(tail -1 models-anlz/$tb/log-$exp.out | cut -f 1)

	input_lem_model=models-segm/$tb/segm_"$op_ep_seg".pth
	# input_anlz_model=models-anlz/$tb/anlz_"$op_ep_anl".pth
	input_anlz_model=models-anlz/$tb/$exp_"$op_ep_anl".pth
	emb_file=models-segm/$tb/emb.pth
	
	qsub -q 'gpu-troja.q' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-anlz/$tb/log-$exp-$mode.out \
	-e models-anlz/$tb/log-$exp-$mode.err \
	wraps/test_anlz.sh $tb $mode "$input_lem_model" "$input_anlz_model" $emb_file $bsearch $tagger_mode $eval_mode
	
	# bash wraps/test_anlz.sh $tb $mode "$input_lem_model" "$input_anlz_model" $emb_file
	

done
