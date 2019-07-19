#!/bin/bash

cd $HOME/universal-lang-tools-playground


rl=$1
il=$2
base=$3
nc=$4
nj=$5
mode=$6

niters=500
order=2


python3 run_pipeline.py -il $il -rl $rl -b $base -it $niters -rc $nj -lm $order \
-nc $nc -j $nj -m $mode > logs/$rl$order-"$il"."$base".$nc.$niters.pipeline 2> logs/$rl$order-"$il"."$base".$nc.$niters.err
