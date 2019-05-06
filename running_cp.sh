python3 run_analizer.py --mode train \
--train_file data/es_ancora/train \
--dev_file data/es_ancora/dev


python3 random_search.py \
--train_file data/es_ancora/train \
--dev_file data/es_ancora/dev \
--epochs 20
