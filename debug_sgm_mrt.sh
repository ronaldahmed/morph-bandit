#!/bin/bash

tbname="es_ancora"
optmzs="adam adadelta"
alphas="0.01 0.001 0.0001"

for optm in $optmzs; do
	for alpha in $alphas; do
		fn=models-segm/$tbname/l1.mrt_optm-"$optm"_alpha-"$alpha"_sample-10/
		echo ":: optm=$optm  alpha=$alpha  sample=10"
		tail -40 $fn/log.out
		echo ""
		echo ""
	done
done