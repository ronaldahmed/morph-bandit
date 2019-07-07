#!/bin/bash

lsrc=$1
src=$2

basefn="$HOME/morph-bandit/l1-multi-emb/log_$lsrc-es"

qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=4G,mem_free=10G,act_mem_free=10G,h_data=15G -p -10 \
-o $basefn.out \
-e $basefn.err \
wraps/run_muse_multi.sh $lsrc $src

# bash wraps/run_muse_multi.sh $lsrc $src
