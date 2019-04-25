#!/bin/bash

src=$1
tgt=$2
dim=100



python3 $HOME/MUSE/unsupervised.py --src_lang en --tgt_lang cs --src_emb emb/$src.vec \
 --tgt_emb emb/$tgt.vec --n_refinement 5 --emb_dim $dim \
 --seed 42 --exp_path emb/multi \
 --exp_id 0 --exp_name $src-$tgt --cuda False \
 --normalize_embeddings center

