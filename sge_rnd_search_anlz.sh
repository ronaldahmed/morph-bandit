#!/bin/bash

tb=$1
beam_size=5

echo $tb

op_ep=$(tail -1 models-segm/$tb/log.out | cut -f 1)

input_model=models-segm/$tb/segm_$op_ep.pth
emb_file=models-segm/$tb/emb.pth

qsub -q 'gpu*' -cwd -l gpu=1,gpu_cc_min3.5=1,gpu_ram=3G,mem_free=10G,act_mem_free=10G,h_data=15G -p -5 \
-o models-anlz/$tb/log-rnd-search.out \
-e models-anlz/$tb/log-rnd-search.err \
wraps/random_search_anlz.sh $tb $input_model $emb_file $beam_size 42

# bash wraps/random_search_anlz.sh $tb $input_model $emb_file $beam_size 42

