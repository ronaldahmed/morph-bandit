#!/bin/bash

# ISO -> UD directories mapper
. iso_ud_mapper.sh

lid=$1
ud_dir=$2
mem_lim=$3

basedir="$(pwd)"
bdir_t2=$basedir/task2

cd baselines-t2/lemming

echo "lang: -$lid-"
echo ":: $ud_dir"

# annotate
./lemming annotate --marmot_model=models/$ud_dir/full/model.marmot \
--lemming_model=models/$ud_dir/full/model.lemming \
--java_heap_limit=$mem_lim \
--input_file=$(ls $bdir_t2/$ud_dir/*-um-dev.conllu) \
--pred_file=$bdir_t2/$ud_dir/pred-dev.conllu


# lemma acc
./lemming accuracy --tag lemma \
--pred_file $bdir_t2/$ud_dir/pred-dev.conllu \
--oracle_file $(ls $bdir_t2/$ud_dir/*-um-dev.conllu)

# msd acc
./lemming accuracy --tag mtag \
--pred_file $bdir_t2/$ud_dir/pred-dev.conllu \
--oracle_file $(ls $bdir_t2/$ud_dir/*-um-dev.conllu)



## ${} -> parameter expansion + commmand substitution
## $() -> just command substitution