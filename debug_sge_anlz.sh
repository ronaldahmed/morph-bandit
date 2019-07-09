#!/bin/bash

batch=$1
mode=$2
exp=$3

for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb ----------------------------"
	# cat models-anlz/$tb/log.err
	cat models-anlz/$tb/log-$exp-$mode.err
	echo ""
	echo ""
done
