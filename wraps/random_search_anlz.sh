#!/bin/bash

tbname=$1
imodel=$2
emb_file=$3
beam_size=$4
seed=$5

cd /home/acosta/morph-bandit/
conda activate morph



python3 random_search_anlz.py --mode train \
--seed $seed \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 50 \
--emb_size 140 \
--embedding_pth $emb_file \
--input_lem_model $imodel \
--beam_size $beam_size \
--rel_prunning 0.3 \
--scheduler \
--gpu