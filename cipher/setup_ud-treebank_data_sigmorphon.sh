#!/bin/sh

set -e


BASEDIR="$HOME/universal-lang-tools-playground"
UD_DIR="$HOME/morph-bandit/2019/task2"
data_dir=$BASEDIR/data
BATCH="$HOME/morph-bandit/data/tbnames-thesis"

echo $BASEDIR



while [ $# -gt 1 ]
do
key="$1"
case $key in
    -td|--tb_dir) # treebank directory
    UD_DIR="$2"
    shift # past argument
    ;;
    -b|--batch) # treebank directory
    BATCH="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift
done 


########################################################################################



while IFS=" " read -r uddir tbname; do
	echo "$tbname"
	mkdir -p $data_dir/$tbname;
  
  grep -v "^#" $UD_DIR/$uddir/$tbname-um-train.conllu | grep -v "^\s*$" | \
  grep -vP "^[0-9]+-[0-9]+" > $data_dir/$tbname/train.conllu

	# cp $UD_DIR/$uddir/$tbname-um-train.conllu $data_dir/$tbname/train.conllu
  
done < $BATCH


grep -v "^#" $HOME/morph-bandit/shk/shp-um-test.conllu | grep -v "^\s*$" | \
grep -vP "^[0-9]+-[0-9]+" > $data_dir/sk/test.conllu