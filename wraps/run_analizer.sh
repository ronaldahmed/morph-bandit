#!/bin/bash

tbname=$1
imodel=$2
emb_file=$3

cd /home/acosta/morph-bandit/
conda activate morph

python3 run_analizer.py --mode train \
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
--batch_size 5 \
--rel_prunning 0.3 \
--gpu

