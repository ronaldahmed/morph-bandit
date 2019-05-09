#!/bin/bash

batch=$1
mode=$2  # dev, test, covered-test

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	# bash wraps/run_analizer.sh $tb models-segm/$tb
	op_ep_seg=$(tail -1 models-segm/$tb/log.out | cut -f 1)

	if [ $op_ep_seg == "0" ]; then
		op_ep_seg="19"
	fi
	if [ $tb == "kpv_ikdp" ]; then
		op_ep_seg="9"
	fi

	op_ep_anl=$(tail -1 models-anlz/$tb/log.out | cut -f 1)

	if [ $op_ep_anl == "0" ]; then
		op_ep_anl="19"
	fi
	if [ $tb == "kpv_lattice" ]; then
		op_ep_anl="16"
	fi

	input_lem_model=models-segm/$tb/segm_"$op_ep_seg".pth
	input_anlz_model=models-anlz/$tb/anlz_"$op_ep_anl".pth
	emb_file=models-segm/$tb/emb.pth
	
	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-anlz/$tb/log-$mode.out \
	-e models-anlz/$tb/log-$mode.err \
	wraps/test_anlz.sh $tb $mode "$input_lem_model" "$input_anlz_model" $emb_file
	

done
