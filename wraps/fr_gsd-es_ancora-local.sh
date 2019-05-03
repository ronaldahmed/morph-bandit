#!/bin/bash

cd /home/ronald/MUSE
python supervised.py --seed 42 --cuda True --normalize_embeddings center --exp_path /home/ronald/morph-bandit/emb/multi/ --exp_name fr_gsd-es_ancora --exp_id fr_gsd-es_ancora --src_lang fr_gsd --tgt_lang es_ancora --emb_dim 100 --dico_train /home/ronald/morph-bandit/dicts/fr-es.0-5000.ops --dico_eval /home/ronald/morph-bandit/dicts/fr-es.5000-6500.ops --src_emb /home/ronald/morph-bandit/emb/fr_gsd.vec --tgt_emb /home/ronald/morph-bandit/emb/es_ancora.vec --n_refinement 20
