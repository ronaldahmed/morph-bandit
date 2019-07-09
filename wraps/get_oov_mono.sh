#!/bin/bash


# uddir=$1
# tbname=$2
batch=$1

mkdir -p emb-mono/

cd $HOME/morph-bandit

while IFS=" " read -r uddir tbname; do
	echo "$uddir - $tbname"
	
	cat 2019/task2/$uddir/"$tbname"-um-train.conllu | \
	grep -vP "^#" | grep -vP "^$" | cut -f 2 | sort | uniq > train.vocab
	cat 2019/task2/$uddir/"$tbname"-um-dev.conllu | \
	grep -vP "^#" | grep -vP "^$" | cut -f 2 | sort | uniq > dev.vocab
	cat 2019/task2/$uddir/"$tbname"-um-covered-test.conllu | \
	grep -vP "^#" | grep -vP "^$" | cut -f 2 | sort | uniq > test.vocab

	python ft_mono_queries.py train.vocab dev.vocab test.vocab > emb-mono/$tbname.queries

	fasttext print-word-vectors emb-mono/$tbname.bin < emb-mono/$tbname.queries > emb-mono/$tbname.oov.vec

	rm train.vocab dev.vocab test.vocab

done < $batch
