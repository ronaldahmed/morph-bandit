#!/bin/bash

lang=$1

qsub -cwd -l mem_free=20G,act_mem_free=20G,h_vmem=30G -p -50 \
-o baselines-t2/lemming/models/$lang.out \
-e baselines-t2/lemming/models/$lang.err \
train_lemming.sh $lang
