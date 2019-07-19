#!/bin/bash

rl=$1
il=$2
base=$3
nc=$4
nj=20

niters=500
order=2

cd $HOME/morph-bandit/cipher
log_dir=$HOME/universal-lang-tools-playground

qsub -cwd -l mem_free=15G,act_mem_free=15G,h_vmem=20G -p -10 \
-o $log_dir/$rl$order-"$il"."$base".$nc.$niters.pipeline \
-e $log_dir/$rl$order-"$il"."$base".$nc.$niters.err \
run_exp.sh $rl $il $base $nc $nj train