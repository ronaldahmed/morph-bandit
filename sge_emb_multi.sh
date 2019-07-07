#!/bin/bash

lsrc=$1
src=$2

cd /home/acosta/personal_work_ms/MUSE

src_emb="/home/acosta/morph-bandit/l1-mono-emb/$src.vec"
tgt_emb="/home/acosta/morph-bandit/l1-mono-emb/es_ancora.vec"

python unsupervised.py --src_lang $lsrc --tgt_lang es \
--src_emb $src_emb --tgt_emb $tgt_emb \
--n_refinement 5 \
--normalize_embeddings \
--seed 42 \
--cuda \
