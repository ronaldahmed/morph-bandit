#!/bin/bash

batch=$1
for tb in $(cut -f 2 -d " " $batch); do
	echo "::$tb -------------------------------------------"
	cat models-segm/$tb/log-dev.err
	echo ""
	echo ""
done
