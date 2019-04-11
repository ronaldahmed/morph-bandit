#!/bin/bash

lang=$1

#source /home/acosta/pytorch-gpu-1.0/bin/activate

qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=20G,act_mem_free=20G,h_data=30G -p -50 \
-o baselines-t2/neural-lemmatizer/models/$lang.out \
-e baselines-t2/neural-lemmatizer/models/$lang.err \
train_neural_lem.sh $lang
