#!/bin/bash

# ISO -> UD directories mapper
. iso_ud_mapper.sh

#langs: en,es,de,cs,tr,shk,[ar,ja]
langs="en es cs tr"
basedir="$(pwd)"

## Subtask1 : lemmatization
bdir_t2=$basedir/task2

# a) get predictions

echo "$bdir_t2"

# lemming

## might run out of memory: FFS!!

cd baselines-t2/lemming
for lid in $langs; do
  echo "lang: -$lid-"
  phdirs="${iso_ud[$lid]}"
  for ud_dir in $phdirs; do
    echo ":: $ud_dir"
    mkdir -p models/$ud_dir
    ./lemming train \
    --ud_data_dir $bdir_t2/$ud_dir \
    --exp_dir models/$ud_dir            #  <- works!, check memory intake

    break
  done

  break
done

# annotate

for lid in $langs; do
  echo "lang: -$lid-"
  phdirs="${iso_ud[$lid]}"
  for ud_dir in $phdirs; do
    ./lemming annotate --marmot_model=models/$ud_dir/model.marmot \
    --lemming_model=models/$ud_dir/model.lemming \
    --input_file=$(bdir_t2/$ud_dir/*-um-dev.conllu) \
    --pred_file=bdir_t2/$ud_dir/pred-dev.conllu

    break
  done

  break
done

# eval 

for lid in $langs; do
  echo "lang: -$lid-"
  phdirs="${iso_ud[$lid]}"
  for ud_dir in $phdirs; do
    ./lemming accuracy --tag lemma \
    --pred_file bdir_t2/$ud_dir/pred-dev.conllu \
    --oracle_file $(bdir_t2/$ud_dir/*-um-dev.conllu)

    break
  done
  
  break
done



## ${} -> parameter expansion + commmand substitution
## $() -> just command substitution