#!/bin/bash

set -e

batch="data/tbnames-thesis"
exp="l1.mrt"
loss="mrt" # "-"
optm="adam"
alpha_q="0.1"
sample_size="10"
batch_size="10" # 128 for MLE
temperature=10.0

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
    -temp|--temp)
    temperature="$2"
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
        outdir=models-segm/$tb/"$exp"_optm-"$optm"_alpha-"$alpha_q"_sample-"$sample_size"_clip-"$clip"_bs-"$batch_size"-
        op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)
		input_model=models-segm/$tb/segm_$op_ep.pth
    fi
	mkdir -p $outdir

    bash \
	wraps/run_lemmatizer.sh \
    -tb $tb -m train --outdir $outdir --exp $exp \
    --loss $loss -optm $optm -a $alpha_q -s $sample_size \
    -bs $batch_size -lr 0.0001 -dp 0 -ilem $input_model \
    -temp $temperature \
    > $outdir/log.err >2 $outdir/log.err

done