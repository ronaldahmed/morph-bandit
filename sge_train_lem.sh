#!/bin/bash

set -e

batch="data/tbnames-thesis"
exp="l1.mrt"
loss="mrt" # "-"
optm="adam"
alpha_q="0.1"
sample_size="10"

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
    if [ $loss == "mle" ];then
        outdir=models-segm/$tb
    else
        outdir=models-segm/$tb/"$exp"_optm-"$optm"_alpha-"$alpha_q"_sample-"$sample_size"
    fi
	mkdir -p $outdir
	

	# qsub -q 'gpu-troja.q' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	# -o models-segm/$tb/log-$exp.out \
	# -e models-segm/$tb/log-$exp.err \
    bash \
	wraps/run_lemmatizer.sh \
    -tb $tb -m train --outdir $outdir --exp $exp \
    --loss $loss -optm $optm -a $alpha_q -s $sample_size \
    -bs 20 -lr 0.0001 -dp 0 

done