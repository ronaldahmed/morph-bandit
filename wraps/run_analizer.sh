#!/bin/bash

tbname=$1
mdir=$2

python3 run_analizer.py --mode train \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 20 \
--batch_size 32 \
--mlp_size 100 \
--emb_size 300 \
--learning_rate 0.0005 \
--dropout 0.1 \
--model_save_dir $mdir \
--scheduler \
--gpu

