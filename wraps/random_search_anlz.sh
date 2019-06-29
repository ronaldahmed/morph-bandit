#!/bin/bash

tbname=$1

cd /home/acosta/morph-bandit/
conda activate morph


python3 random_search_anlz.py --mode train \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 50 \
--scheduler \
--gpu