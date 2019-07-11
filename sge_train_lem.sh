#!/bin/bash

set -e

batch="data/tbnames-thesis"
exp="l1.mle"
loss="mrt"
loss="adam"

while [ $# -gt 1 ]
do
key="$1"
case $key in
    -b|--task)
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
    *)
            # unknown option
    ;;
esac
shift
done 


if [ ${exp: -3} == "mle" ]; then
	loss="mle"
elif [ ${exp: -3} == "mrt" ]; then
	loss="mrt"
fi

mkdir -p models-segm

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	mkdir -p models-segm/$tb
	# bash wraps/run_lemmatizer.sh $tb models-segm/$tb

	qsub -q 'gpu-troja.q' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
	-o models-segm/$tb/log-$exp.out \
	-e models-segm/$tb/log-$exp.err \
	wraps/run_lemmatizer.sh $tb train models-segm/$tb 42 $loss $exp

done