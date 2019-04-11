#!/bin/bash

# ISO -> UD directories mapper
. iso_ud_mapper.sh

langs="en es cs tr ar ja de mt"

basedir="$(pwd)"
bdir_t2=$basedir/task2

cd baselines-t2/lemming

for lid in $langs; do
  echo "lang: -$lid-"
  phdirs="${iso_ud[$lid]}"

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
done
  




## ${} -> parameter expansion + commmand substitution
## $() -> just command substitution