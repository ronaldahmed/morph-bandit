#!/bin/bash

lang=$1
ud_dir=$2

qsub -cwd -l mem_free=100G,act_mem_free=100G,h_vmem=150G -p -50 \
-o baselines-t2/lemming/models/$lang-"$ud_dir".out \
-e baselines-t2/lemming/models/$lang-"$ud_dir".err \
annot_lemming_uddir.sh $lang $ud_dir 100
