#!/bin/bash

cd /home/ronald/MUSE
python supervised.py --seed 42 --cuda True --normalize_embeddings center --exp_path /home/ronald/morph-bandit/emb/multi/ --exp_name es_ancora-en_ewt --exp_id es_ancora-en_ewt --src_lang es_ancora --tgt_lang en_ewt --emb_dim 100 --dico_train /home/ronald/morph-bandit/dicts/es-en.0-5000.ops --dico_eval /home/ronald/morph-bandit/dicts/es-en.5000-6500.ops --src_emb /home/ronald/morph-bandit/emb/es_ancora.vec --tgt_emb /home/ronald/morph-bandit/emb/en_ewt.vec --n_refinement 20
