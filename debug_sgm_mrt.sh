#!/bin/bash

tbname="es_ancora"


for optm in "adam adadelta"; do
	for alpha in "0.01 0.001 0.0001"; do
		fn=models-segm/$tbname/l1.mrt_optm-"$optm"_alpha-$alpha_sample-10/
		echo ":: optm=$optm  alpha=$alpha  sample=10"
		tail -40 $fn/log.out
		echo ""
		echo ""
	done
done