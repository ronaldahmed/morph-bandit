#!/bin/bash

batch=$1
for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb ----------------------------"
	tail -10 models-anlz/$tb/log.out
	echo ""
	echo ""
done
