#!/bin/bash

tbname=$1
mode=$2
imodel=$3
emb_file=$4
beam_size=$5
seed=$6
tagger_mode=$7
exp_id=$8

echo $emb_file
echo $seed

cd /home/acosta/morph-bandit/
conda activate morph

python3 run_analizer.py --mode $mode \
--seed $seed \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 100 \
--batch_size 24 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.0001 \
--dropout 0.05 \
--embedding_pth $emb_file \
--input_lem_model $imodel \
--model_save_dir models-anlz/$tbname \
--scheduler \
--beam_size $beam_size \
--rel_prunning 0.3 \
--tagger_mode $tagger_mode \
--exp_id $exp_id \
--gpu

