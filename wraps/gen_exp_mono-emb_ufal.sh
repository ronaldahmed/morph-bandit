#!/bin/bash

tbname=$1
input=$2
njobs=20

outname="wraps/ft-$tbname"

echo "#!/bin/bash" > $outname
echo "" >> $outname
echo "qsub -cwd -l mem_free=15G,act_mem_free=15G,h_vmem=22G -p -50 -pe smp $njobs \ " >> $outname
echo "-o $outname.out \ " >> $outname
echo "-e $outname.err \ " >> $outname
echo "$HOME/fastText/fasttext skipgram -minCount 1 -input $input -output emb/$tbname -thread $njobs " >> $outname
