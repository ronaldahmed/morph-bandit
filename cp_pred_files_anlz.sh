#!/bin/bash

batch=$1
mode=$2
exp=$3
mkdir -p models_pred

for tb in $(cut -f 2 -d " " $batch); do
	echo $tb
	cp data/$tb/$mode.$exp.conllu.pred models_pred/$tb-um-$mode.conllu.$exp.pred
done