#!/bin/bash

set -e

tbname=$1

mkdir -p models-segm/$tbname
mkdir -p models-anlz/$tbname
scp acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-segm/$tbname/log.out models-segm/$tbname
scp acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-anlz/$tbname/log.out models-anlz/$tbname

op_ep_seg=$(tail -1 models-segm/$tbname/log.out | cut -f 1)
op_ep_anl=$(tail -1 models-anlz/$tbname/log.out | cut -f 1)

scp acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-segm/$tbname/segm_"$op_ep_seg".pth models-segm/$tbname
scp acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-segm/$tbname/emb.pth models-segm/$tbname

scp acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-anlz/$tbname/anlz_"$op_ep_anl".pth models-anlz/$tbname

scp -r acosta@freki.ms.mff.cuni.cz:~/morph-bandit/data/$tbname data

# scp -r acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-segm/$tbname models-segm
# scp -r acosta@freki.ms.mff.cuni.cz:~/morph-bandit/models-anlz/$tbname models-anlz

