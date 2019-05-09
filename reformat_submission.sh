#!/bin/bash

dirname="CHARLES-MALTA-01-2"
mkdir -p $dirname

ud_tb_names=data/uddir_tbname

for line in $(cat "$ud_tb_names"); do
	uddir=$(cut -f 1 -d " " $line)
	tbname=$(cut -f 2 -d " " $line)
	mkdir -p $dirname/$uddir
	cp data/$tbname/test.anlz.conllu.pred $dirname/$uddir/$tbname-um-test.output
done
