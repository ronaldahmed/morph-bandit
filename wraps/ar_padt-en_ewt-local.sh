#!/bin/bash

cd /home/ronald/MUSE
python supervised.py --seed 42 --cuda True --normalize_embeddings center --exp_path /home/ronald/morph-bandit/emb/multi/ --exp_name ar_padt-en_ewt --exp_id ar_padt-en_ewt --src_lang ar_padt --tgt_lang en_ewt --emb_dim 100 --dico_train /home/ronald/morph-bandit/dicts/ar-en.0-5000.ops --dico_eval /home/ronald/morph-bandit/dicts/ar-en.5000-6500.ops --src_emb /home/ronald/morph-bandit/emb/ar_padt.vec --tgt_emb /home/ronald/morph-bandit/emb/en_ewt.vec --n_refinement 20
