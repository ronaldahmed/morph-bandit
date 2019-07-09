#!/bin/bash


# uddir=$1
# tbname=$2
batch=$1

mkdir -p emb-mono/

cd $HOME/morph-bandit

while IFS=" " read -r uddir tbname; do
	echo "$uddir - $tbname"
	
	cat 2019/task2/$uddir/"$tbname"-um-train.conllu | \
	grep -vP "^#" | cut -f 2 | \
	python conllu_to_txt.py > emb-mono/"$tbname".txt

	fasttext skipgram -input emb-mono/"$tbname".txt -output emb-mono/"$tbname"

done < $batch
