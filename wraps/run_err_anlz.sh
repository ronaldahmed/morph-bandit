#!/bin/bash

tbname=$1

cd /home/acosta/morph-bandit/
conda activate morph

python3 error_analysis.py --mode dev \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--embedding_pth foo \
--input_lem_model foo \
--src_ref "lem" \
--tgt_ref "anlz.fine-seq"

--tagger_mode "fine-seq"

