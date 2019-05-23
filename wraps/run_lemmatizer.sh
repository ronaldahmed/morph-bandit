#!/bin/bash

tbname=$1
mdir=$2

cd /home/acosta/morph-bandit/
conda activate morph

python3 run_lemmatizer.py --mode train \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 30 \
--batch_size 128 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.00069 \
--dropout 0.19 \
--model_save_dir $mdir \
--scheduler \
--gpu

