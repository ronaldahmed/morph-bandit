#!/bin/bash

# set -e

tbname="shk_cuni"
mode=train
fold=0
mdir="models-segm/shk_cuni"
seed=42
exp_id="l1.mrt"
loss="mrt" # "-"
optm="adam"
alpha_q="0.1"
sample_size="20"
batch_size=128
learning_rate="0.00069"
dropout="0.19"
input_model="-"
clip="0"
temperature=10.0
beam_size=-1
GPU=0

while [ $# -gt 1 ]
do
key="$1"
case $key in
    -tb|--tbname)
    tbname="$2"
    shift # single-task
    ;;
	-m|--mode)
    mode="$2"
    shift # single-task
    ;;
	-od|--outdir)
    mdir="$2"
    shift # single-task
    ;;
    -l|--loss)
    loss="$2"
    shift # pretrained model
    ;;
    -o|--optm)
    optm="$2"
    shift # pretrained model
    ;;
    -a|--alpha_q)
    alpha_q="$2"
    shift # pretrained model
    ;;
    -s|--sample)
    sample_size="$2"
    shift # pretrained model
    ;;
    -bs|--batch_size)
    batch_size="$2"
    shift # pretrained model
    ;;
    -lr|--lr)
    learning_rate="$2"
    shift # pretrained model
    ;;
    -dp|--dropout)
    dropout="$2"
    shift # pretrained model
    ;;
    -ilem|--ilem)
    input_model="$2"
    shift # pretrained model
    ;;
    -c|--clip)
    clip="$2"
    shift # pretrained model
    ;;
    -temp|--temp)
    temperature="$2"
    shift # pretrained model
    ;;
    -beam|--beam)
    beam_size="$2"
    shift # pretrained model
    ;;
    -f|--fold)
    fold="$2"
    shift # pretrained model
    ;;
    -gpu|--gpu)
    GPU="$2"
    shift # pretrained model
    ;;
    *)
            # unknown option
    ;;
esac
shift
done 


cd ..

mkdir -p $mdir/fold.${fold}

exp_id=lem.${loss}.${fold}

CUDA_VISIBLE_DEVICES=${GPU} python3 run_lemmatizer.py --mode $mode \
--seed $seed \
--train_file data/$tbname/train.${fold} \
--dev_file data/$tbname/test.${fold} \
--test_file data/$tbname/test.${fold} \
--input_model $input_model \
--epochs 20 \
--batch_size $batch_size \
--mlp_size 100 \
--emb_size 140 \
--learning_rate $learning_rate \
--dropout $dropout \
--clip $clip \
--model_save_dir $mdir/fold.${fold} \
--scheduler \
--lem_loss $loss \
--exp_id $exp_id \
--alpha_q $alpha_q \
--sample_space_size $sample_size \
--lem_optm $optm \
--temperature $temperature \
--beam_size $beam_size \
--rel_prunning 0.3 \
--gpu > $mdir/fold.${fold}/${exp_id}.log 2> $mdir/fold.${fold}/${exp_id}.err

