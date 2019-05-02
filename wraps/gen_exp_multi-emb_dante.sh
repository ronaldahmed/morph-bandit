#!/bin/bash

src=$1
tgt=$2

src_lang=${src:0:2}
tgt_lang=${tgt:0:2}

basedir=/users/cborg/rcardenas/morph-bandit
outfile=$src-$tgt-dante.sh

echo "#!/bin/bash" > $outfile
echo "#SBATCH --job-name=$src-$tgt" >> $outfile
echo "#SBATCH --output=$basedir/emb/multi/$src-$tgt.log" >> $outfile
echo "#SBATCH --nodes=1" >> $outfile
echo "#SBATCH --ntasks=20" >> $outfile
echo "#SBATCH --mem=40GB" >> $outfile
echo "#SBATCH --gres=gpu:1" >> $outfile
echo "#SBATCH --time=100:00:00" >> $outfile
echo "" >> $outfile

echo "source /users/cborg/.bashrc" >> $outfile
echo "conda init bash" >> $outfile
echo "conda activate sopa" >> $outfile
echo "" >> $outfile

echo "cd /users/cborg/rcardenas/MUSE" >> $outfile

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