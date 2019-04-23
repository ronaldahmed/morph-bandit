#!/bin/bash

treebank=$1
input=$2
outfile=mono-"$treebank"-dante.sh
njobs=30

echo "#!/bin/bash" > $outfile
echo "" >> $outfile

echo "#SBATCH --job-name=mono_$treebank" >> $outfile
echo "#SBATCH --output=/users/cborg/rcardenas/morph-bandit/wraps/mono-$treebank.log" >> $outfile
echo "#SBATCH --nodes=1" >> $outfile
echo "#SBATCH --ntasks=$njobs" >> $outfile
echo "#SBATCH --mem=50GB" >> $outfile
echo "#SBATCH --time=100:00:00" >> $outfile
echo "" >> $outfile

echo "source /users/cborg/.bashrc" >> $outfile
echo "conda init bash" >> $outfile
echo "conda activate sopa" >> $outfile
echo "cd /users/cborg/rcardenas/morph-bandit/" >> $outfile
echo "" >> $outfile

echo "/users/cborg/rcardenas/fastText/fasttext skipgram -minCount 1 -input $input -output emb/$treebank -thread $njobs" >> $outfile
