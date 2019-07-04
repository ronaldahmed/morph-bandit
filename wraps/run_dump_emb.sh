#!/bin/bash

tbname=foo

cd /home/acosta/morph-bandit/
conda activate morph

python3 dump_e-lem_vec.py --mode dev \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--embedding_pth foo \
--input_lem_model foo \
--src_ref "anlz" \
--tgt_ref "anlz.fine-seq"

