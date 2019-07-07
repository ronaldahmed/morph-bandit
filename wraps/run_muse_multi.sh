#!/bin/bash


lsrc=$1
src=$2

conda activate morph

cd /home/acosta/personal_work_ms/MUSE

basedir="/home/acosta/morph-bandit/l1-multi-emb"
src_emb="/home/acosta/morph-bandit/l1-mono-emb/$src.vec"
tgt_emb="/home/acosta/morph-bandit/l1-mono-emb/es_ancora.vec"

python unsupervised.py --src_lang $lsrc --tgt_lang es \
--src_emb $src_emb --tgt_emb $tgt_emb \
--emb_dir 140 \
--n_refinement 5 \
--normalize_embeddings center \
--seed 42 \
--exp_path $basedir \
--exp_name "$lsrc-es" \
--export pth \
--cuda True
