#!/bin/bash

set -e


LM=2            # LM order
CHANNEL=""
INPUT=""
OUTPUT="output"

MODE="train" # [train,test]
ROM="false"
IL="xx"     # incident language
PL="en"     # parent language(s), comman separated
LOWER="True"
BASELINE="brown"
clustdir="exp_cipher/brown-500"
NCLUSTERS=500
EXP_DIR="exp-cipher"


cd $HOME/universal-cipher-pos-tagging
DATADIR="$HOME/universal-cipher-pos-tagging/lm_data"
lang="sk"

TAGSET="um"

grep -v "^#" $HOME/morph-bandit/shk/shp-um-test.conllu | grep -v "^\s*$" | \
grep -vP "^[0-9]+-[0-9]+" > $DATADIR/$lang/test.conllu


python3 src/code/conllu2txt.py -i $DATADIR/$lang/test.conllu -c 1 -lid \
> $DATADIR/$lang/test.raw

src/code/replace-unicode-punctuation.perl < $DATADIR/$lang/test.raw > $DATADIR/$lang/test.clean

python3 src/code/filter_lowfreq.py -i $DATADIR/$lang/test.clean -m eval -low $LOWER -t 1 -v $DATADIR/$lang/vocab.sk

cp $DATADIR/$lang/test.clean.filt $DATADIR/$lang/test.true



python3 src/code/conllu2txt.py -i $DATADIR/$lang/test.conllu \
-m ch -c 5 -tb $TAGSET > $DATADIR/$lang/test.upos.ch

python3 src/code/conllu2txt.py -i $DATADIR/$lang/test.conllu \
-m tag -c 5 -tb $TAGSET > $DATADIR/$lang/test.upos


bash src/code/tag_with_clusters.sh -b $BASELINE -n $NCLUSTERS -i $DATADIR/$lang/test.clean.filt -e $EXP_DIR
