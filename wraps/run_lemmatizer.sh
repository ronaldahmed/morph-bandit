#!/bin/bash

tbname=$1
mode=$2
mdir=$3
seed=$4
loss=$5
exp_id=$6

cd /home/acosta/morph-bandit/
conda activate morph

python3 run_lemmatizer.py --mode $mode \
--seed $seed \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 20 \
--batch_size 128 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.00069 \
--dropout 0.19 \
--model_save_dir $mdir \
--scheduler \
--lem_loss $loss \
--exp_id $exp_id \
--gpu

