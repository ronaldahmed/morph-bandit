#!/bin/bash
#SBATCH --job-name=it_isdt-es_ancora
#SBATCH --output=/users/cborg/rcardenas/morph-bandit/emb/multi/it_isdt-es_ancora.log
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

source /users/cborg/.bashrc
conda init bash
conda activate sopa

cd /users/cborg/rcardenas/MUSE
python supervised.py --seed 42 --cuda --normalize_embeddings --exp_path emb/multi/ --exp_name it_isdt-es_ancora --exp_id it_isdt-es_ancora --src_lang it_isdt --tgt_lang es_ancora --emb_dim 100 --dico_train /users/cborg/rcardenas/morph-bandit/dicts/it-es.0-5000.ops --dico_eval /users/cborg/rcardenas/morph-bandit/dicts/it-es.5000-6500.ops --src_emb /users/cborg/rcardenas/morph-bandit/emb/it_isdt.vec --tgt_emb /users/cborg/rcardenas/morph-bandit/emb/es_ancora.vec --n_refinement 5
