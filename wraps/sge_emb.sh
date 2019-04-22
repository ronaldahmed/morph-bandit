#!/bin/bash

treebank=$1

qsub -pe smp 50 -cwd -l mem_free=40G,act_mem_free=40G,h_data=60G -p -50 \
-o emb/logs/$treebank.out \
-e emb/logs/$treebank.err \
wraps/train_emb_ft.sh $treebank