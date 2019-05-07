#!/bin/bash

tbname=$1
mode=$2
imodel=$3

cd /home/acosta/morph-bandit/
conda activate morph

if [ $mode == "covered-test" ]||[ $mode == "test" ]; then
	python3 run_lemmatizer.py --mode $mode \
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

elif [ $mode == "dev" ]; then
	python3 run_lemmatizer.py --mode $mode \
	--train_file data/$tbname/train \
	--dev_file data/$tbname/dev \
	--epochs 20 \
	--batch_size 128 \
	--mlp_size 100 \
	--emb_size 140 \
	--learning_rate 0.00069 \
	--dropout 0.19 \
	--scheduler \
	--input_model $imodel \
	--gpu


fi