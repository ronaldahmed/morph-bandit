#!/bin/bash

batch=$1

cd $HOME/universal-lang-tools-playground

for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb ----------------------------"
	# cat models-anlz/$tb/log.err
	cat logs/"$tb"2-sk.brown.500.500.err | grep -vP "^\s*$"
	echo ""
	echo ""
done
