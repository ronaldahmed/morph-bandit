#!/bin/bash

tbname="es_ancora"
# optmzs="adam adadelta"
optmzs="adadelta"
# alphas="0.01 0.001 0.0001"
alphas="0.001"
clips="0 1 10 100 1000"

for optm in $optmzs; do
	for alpha in $alphas; do
		for clip in $clips; do
			fn=models-segm/$tbname/l1.mrt.warm_optm-"$optm"_alpha-"$alpha"_sample-10_clip-"$clip"/
			echo ":: optm=$optm  alpha=$alpha  sample=10  clip=$clip"
			tail -40 $fn/log.err
			echo ""
			echo ""
		done
	done
done