#!/bin/bash

lang=$1
ud_dir=$2

qsub -cwd -l mem_free=40G,act_mem_free=40G,h_vmem=600G -p -50 \
-o baselines-t2/lemming/models/$lang-"$ud_dir".out \
-e baselines-t2/lemming/models/$lang-"$ud_dir".err \
train_lemming_ud_dir.sh $lang $ud_dir 40
