#!/bin/bash

set -e

batch="data/tbnames-thesis"
exp="l1.mrt"
loss="mrt" # "-"
optm="adam"
alpha_q="0.05"
sample_size="20"
batch_size="10" # 128 for MLE
clip=0
learning_rate="1e-4"

while [ $# -gt 1 ]
do
key="$1"
case $key in
    -b|--batch)
    batch="$2"
    shift # single-task
    ;;
    -e|--exp)
    exp="$2"
    shift # gpu 
    ;;
    -l|--loss)
    loss="$2"
    shift # pretrained model
    ;;
    -o|--optm)
    optm="$2"
    shift # pretrained model
    ;;
    -a|--alpha_q)
    alpha_q="$2"
    shift # pretrained model
    ;;
    -s|--sample)
    sample_size="$2"
    shift # pretrained model
    ;;
    -c|--clip)
    clip="$2"
    shift # pretrained model
    ;;
    -bs|--bs)
    batch_size="$2"
    shift # pretrained model
    ;;
    *)
            # unknown option
    ;;
esac
shift
done 

if [ $loss == "-" ]; then
    if [ ${exp: -3} == "mle" ]; then
    	loss="mle"
    elif [ ${exp: -3} == "mrt" ]; then
    	loss="mrt"
    fi
fi


for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
    outdir=""
	input_model="-"
    if [ $loss == "mle" ];then
        outdir=models-segm/$tb
    elif [ $loss == "mrt" ]; then
        outdir=models-segm/$tb/"$exp"_optm-"$optm"_alpha-"$alpha_q"_sample-"$sample_size"_clip-"$clip"_bs-"$batch_size"
        op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)
		input_model=models-segm/$tb/segm_$op_ep.pth
    fi
	mkdir -p $outdir

    # 'gpu-troja.q'
    # bash \
	qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=8G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o $outdir/log.out \
	-e $outdir/log.err \
	wraps/run_lemmatizer.sh \
    -tb $tb -m train --outdir $outdir --exp $exp \
    --loss $loss -optm $optm -a $alpha_q -s $sample_size \
    -bs $batch_size -lr $learning_rate -dp 0 -ilem $input_model -c $clip 

done