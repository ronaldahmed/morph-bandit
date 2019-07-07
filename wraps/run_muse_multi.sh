#!/bin/bash


lsrc=$1
src=$2

conda activate morph

cd $HOME/MUSE

basedir="$HOME/morph-bandit/l1-multi-emb"
src_emb="$HOME/morph-bandit/l1-mono-emb/$src.vec"
tgt_emb="$HOME/morph-bandit/l1-mono-emb/es_ancora.vec"

python unsupervised.py --src_lang $lsrc --tgt_lang es \
--src_emb $src_emb --tgt_emb $tgt_emb \
--emb_dim 140 \
--n_refinement 5 \
--normalize_embeddings center \
--seed 42 \
--exp_path $basedir \
--exp_name "$lsrc-es" --exp_id "$lsrc-es" \
--export pth \
--cuda True
