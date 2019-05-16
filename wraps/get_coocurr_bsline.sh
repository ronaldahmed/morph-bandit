#!/bin/bash

tbname=$1
mode=$2
lemmodel=$3
anlmodel=$4
emb_file=$5

cd /home/acosta/morph-bandit/
conda activate morph

python3 get_weighted_coocrr_bselines.py --mode $mode \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 100 \
--batch_size 24 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.0001 \
--dropout 0.05 \
--scheduler \
--embedding_pth $emb_file \
--input_lem_model $lemmodel \
--input_model $anlmodel \
--gpu
