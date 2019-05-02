#!/bin/bash

src=$1
tgt=$2

src_lang=${src:0:2}
tgt_lang=${tgt:0:2}

basedir=$HOME/morph-bandit
outfile=$src-$tgt-local.sh

echo "#!/bin/bash" > $outfile
echo "" >> $outfile


echo "cd $HOME/MUSE" >> $outfile

run_com="
python supervised.py --seed 42 --cuda True --normalize_embeddings center \
--exp_path $basedir/emb/multi/ --exp_name $src-$tgt --exp_id $src-$tgt \
--src_lang $src --tgt_lang $tgt \
--emb_dim 100 \
--dico_train $basedir/dicts/$src_lang-$tgt_lang.0-5000.ops \
--dico_eval $basedir/dicts/$src_lang-$tgt_lang.5000-6500.ops \
--src_emb $basedir/emb/$src.vec \
--tgt_emb $basedir/emb/$tgt.vec \
--n_refinement 5
"

echo $run_com >> $outfile