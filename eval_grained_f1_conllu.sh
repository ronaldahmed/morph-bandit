#!/bin/bash

batch=$1

while IFS=" " read -r uddir tb; do
	echo $tb
	gold_fn=2019/task2/$uddir/$tb-um-dev.conllu
	pred_fn=baseline_pred/$tb-um-dev.conllu.baseline.pred
	feats_vocab_fn=data/$tb/feats.vocab.pickle

	python eval_grained_f1_conllu.py $gold_fn $pred_fn $feats_vocab_fn
done < $batch
