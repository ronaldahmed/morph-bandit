#!/bin/bash

dirname="CHARLES-MALTA-01-2"
mkdir -p $dirname

ud_tb_names=data/uddir_tbname

#for line in $(cat "$ud_tb_names"); do
while read line; do
	uddir=$(echo $line | cut -f 1 -d " ")
	tbname=$(echo $line | cut -f 2 -d " ")
	mkdir -p $dirname/$uddir
	cp data/$tbname/test.anlz.conllu.pred $dirname/$uddir/$tbname-um-test.output
done < $ud_tb_names
