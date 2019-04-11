#!/bin/bash

# ISO -> UD directories mapper
. iso_ud_mapper.sh

lid=$1
ud_dir=$2
mem_limit=$3

basedir="$(pwd)"

## Subtask1 : lemmatization
bdir_t2=$basedir/task2

echo "$bdir_t2"

# lemming

## might run out of memory: FFS!!

cd baselines-t2/lemming
echo "lang: -$lid-"
echo ":: $ud_dir"

mkdir -p models/$ud_dir
./lemming train \
--java_heap_limit=$mem_limit \
--ud_data_dir $bdir_t2/$ud_dir \
--exp_dir models/$ud_dir            #  <- works!, check memory intake


# annotate
./lemming annotate --marmot_model=models/$ud_dir/full/model.marmot \
--lemming_model=models/$ud_dir/full/model.lemming \
--java_heap_limit=$mem_limit \
--input_file=$(ls $bdir_t2/$ud_dir/*-um-dev.conllu) \
--pred_file=$bdir_t2/$ud_dir/pred-dev.conllu



## ${} -> parameter expansion + commmand substitution
## $() -> just command substitution