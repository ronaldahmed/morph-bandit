#!/bin/bash

tbname=$1
mode=$2
imodel=$3
emb_file=$4
beam_size=5
seed=$6

cd /home/acosta/morph-bandit/
conda activate morph

python3 run_analizer.py --mode $mode \
--seed $seed \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 100 \
--batch_size 40 \
--op_enc_size 10 \
--w_enc_size 40 \
--w_mlp_size 150 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.012123789360922501 \
--clip 0.38664772092347766 \
--dropout 0.07191061327716046 \
--embedding_pth $emb_file \
--input_lem_model $imodel \
--model_save_dir models-anlz/$tbname \
--scheduler \
--beam_size $beam_size \
--rel_prunning 0.3 \
--gpu