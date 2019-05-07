#!/bin/bash

tbname=$1
imodel=$2

cd /home/acosta/morph-bandit/
conda activate morph

python3 run_analizer.py --mode covered-test \
--train_file data/$tbname/train \
--test_file data/$tbname/test \
--epochs 20 \
--batch_size 128 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.00069 \
--dropout 0.19 \
--scheduler \
--input_model $imodel \
--gpu

