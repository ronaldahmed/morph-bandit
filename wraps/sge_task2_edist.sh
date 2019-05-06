#!/bin/bash

njobs=$1

cd /home/acosta
conda activate morph

qsub -pe smp $njobs -cwd -l mem_free=2G,act_mem_free=2G,h_data=4G -p -50 \
-o data/task2.log \
-e data/task2.err \
wraps/run_task2_edist.sh $njobs
