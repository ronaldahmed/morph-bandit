#!/bin/bash
#SBATCH --job-name=random_search
#SBATCH --output=/users/cborg/rcardenas/morph-bandit/models/rnd_search-dante.log
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

source /users/cborg/.bashrc
conda init bash
conda activate morph

cd /users/cborg/rcardenas/morph-bandit


python3 random_search.py \
--train_file data/es_ancora/train \
--dev_file data/es_ancora/dev \
--epochs 10 \
--gpu
