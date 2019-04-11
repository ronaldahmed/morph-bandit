#!/bin/bash

# ISO -> UD directories mapper
. iso_ud_mapper.sh

lid=$1
basedir="$(pwd)"

## Subtask1 : lemmatization
bdir_t2=$basedir/task2
phdirs="${iso_ud[$lid]}"

# a) get predictions

echo "$bdir_t2"

# lemming

## might run out of memory: FFS!!

cd baselines-t2/lemming
echo "lang: -$lid-"

for ud_dir in $phdirs; do
  echo ":: $ud_dir"
  mkdir -p models/$ud_dir
  ./lemming train \
  --ud_data_dir $bdir_t2/$ud_dir \
  --exp_dir models/$ud_dir            #  <- works!, check memory intake
done


# annotate
for ud_dir in $phdirs; do
  echo ":: $ud_dir"
  ./lemming annotate --marmot_model=models/$ud_dir/full/model.marmot \
  --lemming_model=models/$ud_dir/full/model.lemming \
  --input_file=$(ls $bdir_t2/$ud_dir/*-um-dev.conllu) \
  --pred_file=$bdir_t2/$ud_dir/pred-dev.conllu
done


# eval 
for ud_dir in $phdirs; do
  echo ":: $ud_dir"
  # lemma acc
  ./lemming accuracy --tag lemma \
  --pred_file $bdir_t2/$ud_dir/pred-dev.conllu \
  --oracle_file $(ls $bdir_t2/$ud_dir/*-um-dev.conllu)
  # msd acc
  ./lemming accuracy --tag mtag \
  --pred_file $bdir_t2/$ud_dir/pred-dev.conllu \
  --oracle_file $(ls $bdir_t2/$ud_dir/*-um-dev.conllu)
done
  



## ${} -> parameter expansion + commmand substitution
## $() -> just command substitution