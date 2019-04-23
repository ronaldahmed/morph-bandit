#!/bin/bash

#SBATCH --job-name=mono_es_ancora
#SBATCH --output=/users/cborg/rcardenas/morph-bandit/wraps/ft-emb_es_ancora.log
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=80GB
#SBATCH --time=100:00:00

source /users/cborg/.bashrc
conda init bash
conda activate sopa
cd /users/cborg/rcardenas/morph-bandit/

/users/cborg/rcardenas/fastText/fasttext skipgram -minCount 1 -input ../2019/task2/UD_Spanish-AnCora/es_ancora-um-train.conllu -output emb/es_ancora -thread 40
