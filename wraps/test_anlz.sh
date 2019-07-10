#!/bin/bash

tbname=$1
mode=$2
lemmodel=$3
anlmodel=$4
emb_file=$5
beam_size=$6
tagger_mode=$7
exp_id=$8

cd /home/acosta/morph-bandit/
conda activate morph

if [ $mode == "covered-test" ]||[ $mode == "test" ]; then
	python3 run_analizer.py --mode $mode \
	--train_file data/$tbname/train \
	--test_file data/$tbname/test \
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
	--dump_ops \
	--beam_size $beam_size \
	--rel_prunning 0.3 \
	--tagger_mode $tagger_mode \
	--exp_id $exp_id \
	--gpu

elif [ $mode == "dev" ]; then
	python3 run_analizer.py --mode $mode \
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
	--dump_ops \
	--beam_size $beam_size \
	--rel_prunning 0.3 \
	--tagger_mode $tagger_mode \
	--exp_id $exp_id \
	--gpu

fi
