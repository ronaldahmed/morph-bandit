#!/bin/bash

set -e

tbname="-"
mode=train
mdir=""
seed=42
exp_id="l1.mrt"
loss="mrt" # "-"
optm="adam"
alpha_q="0.1"
sample_size="10"
batch_size=128
learning_rate="0.00069"
dropout="0.19"

while [ $# -gt 1 ]
do
key="$1"
case $key in
    -tb|--tbname)
    tbname="$2"
    shift # single-task
    ;;
	-m|--mode)
    mode="$2"
    shift # single-task
    ;;
	-od|--outdir)
    mdir="$2"
    shift # single-task
    ;;
    -e|--exp)
    exp_id="$2"
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
    -bs|--batch_size)
    batch_size="$2"
    shift # pretrained model
    ;;
    -lr|--lr)
    learning_rate="$2"
    shift # pretrained model
    ;;
    -dp|--dropout)
    dropout="$2"
    shift # pretrained model
    ;;
    *)
            # unknown option
    ;;
esac
shift
done 


cd $HOME/morph-bandit/
conda activate morph

CUDA_LAUNCH_BLOCKING=1 python3 run_lemmatizer.py --mode $mode \
--seed $seed \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 20 \
--batch_size $batch_size \
--mlp_size 100 \
--emb_size 140 \
--learning_rate $learning_rate \
--dropout $dropout \
--model_save_dir $mdir \
--scheduler \
--lem_loss $loss \
--exp_id $exp_id \
--alpha_q $alpha_q \
--sample_space_size $sample_size \
--lem_optm $optm \
--gpu

