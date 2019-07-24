#!/bin/bash

tb=$1
exp="ml1-a1-bundle"

op_ep_seg=$(tail -1 ../thesis-files/models-segm/$tb/log.out | cut -f 1)
input_lem_model=../thesis-files/models-segm/$tb/segm_"$op_ep_seg".pth
op_ep_anl=""
input_anlz_model=""
emb_file=""

op_ep_anl=$(tail -1 ../thesis-files/models-anlz/$tb/log-$exp.out | cut -f 1)
input_anlz_model=../thesis-files/models-anlz/$tb/"$exp"_"$op_ep_anl".pth

lang_name=${tb:0:2}


if [ $lang_name == "es" ]; then
	emb_file=../thesis-files/l1-multi-emb/cs-es/cs-es/vectors-es.pth
else
	emb_file=../thesis-files/l1-multi-emb/"$lang_name"-es/"$lang_name"-es/vectors-"$lang_name".pth
fi

python3 get_weighted_coocrr_bselines.py \
--mode dev \
--seed 42 \
--train_file data/$tb/train \
--dev_file data/$tb/dev \
--epochs 100 \
--batch_size 24 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.0001 \
--dropout 0.05 \
--embedding_pth $emb_file \
--input_lem_model ${input_lem_model} \
--input_model ${input_anlz_model} \
--tagger_mode bundle \
--exp_id $exp \
--gpu
