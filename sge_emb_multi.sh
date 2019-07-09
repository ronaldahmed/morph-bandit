#!/bin/bash

lsrc=$1
src=$2

basefn="$HOME/morph-bandit/l1-multi-emb/log_$lsrc-es"

qsub -q 'gpu-troja.q' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=8G,act_mem_free=8G,h_data=10G -p -10 \
-o $basefn.out \
-e $basefn.err \
wraps/run_muse_multi.sh $lsrc $src

# bash wraps/run_muse_multi.sh $lsrc $src
