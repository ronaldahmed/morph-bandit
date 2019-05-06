#!/bin/bash

cd /home/ronald/MUSE
python supervised.py --seed 42 --cuda True --normalize_embeddings center --exp_path /home/ronald/morph-bandit/emb/multi/ --exp_name it_isdt-es_ancora --exp_id it_isdt-es_ancora --src_lang it_isdt --tgt_lang es_ancora --emb_dim 100 --dico_train /home/ronald/morph-bandit/dicts/it-es.0-5000.ops --dico_eval /home/ronald/morph-bandit/dicts/it-es.5000-6500.ops --src_emb /home/ronald/morph-bandit/emb/it_isdt.vec --tgt_emb /home/ronald/morph-bandit/emb/es_ancora.vec --n_refinement 20
