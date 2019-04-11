#!/bin/bash

## Neural Lemmatizer

# ISO -> UD directories mapper
. iso_ud_mapper.sh

lid=$1
basedir="$(pwd)"

## Subtask1 : lemmatization
bdir_t2=$basedir/task2
phdirs="${iso_ud[$lid]}"


cd baselines-t2/neural-lemmatizer
echo "lang: -$lid-"

arch=hmm
for ud_dir in $phdirs; do
	echo "$ud_dir"
	mkdir -p models/$ud_dir
	python3 src/train.py \
	    --dataset sigmorphon19task2 \
	    --train=$(ls $bdir_t2/$ud_dir/*-um-train.conllu) \
	    --dev=$(ls $bdir_t2/$ud_dir/*-um-dev.conllu) \
	    --model models/$ud_dir/$lid --seed 0 \
	    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
	    --src_layer 2 --trg_layer 1 --max_norm 5 \
	    --epochs 50 \
	    --arch $arch --estop 1e-8 --bs 20 --mono --bestacc
	    # log: ep 50
done