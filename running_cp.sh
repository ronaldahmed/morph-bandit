python3 run_analizer.py --mode train \
--train_file data/es_ancora/train \
--dev_file data/es_ancora/dev


python3 random_search.py \
--train_file data/es_ancora/train \
--dev_file data/es_ancora/dev \
--epochs 20

export tbname="fi_tdt"
python3 run_analizer.py --mode train \
--train_file data/$tbname/train \
--dev_file data/$tbname/dev \
--epochs 20 \
--batch_size 128 \
--mlp_size 100 \
--emb_size 140 \
--learning_rate 0.00069 \
--dropout 0.19 \
--model_save_dir models-segm/$tbname \
--scheduler \
--gpu





from utils import map_ud_folders

mapper = map_ud_folders()
with open("data/uddir_tbname",'w') as outfile:
	for uddir,tbname in mapper.items():
		print(uddir,tbname,file=outfile)


from utils import map_ud_folders

mapper = map_ud_folders()

list_data = list(mapper.items())
nd = len(list_data)
to_add = (20 - (nd % 20)) % 20
list_data += [(-1,-1)]*to_add
splitted = np.split(np.array(list_data),6)
# splitted[-1] = [x for x in splitted[-1] if x[0]!=-1 and x[1]!=-1]
splitted[-1] = splitted[-1][:7]


for i,batch in enumerate(splitted[:-1]):
	pre = ["%s %s"%(x,y) for x,y in batch]
	open("data/tbnames-"+str(i),'w').write("\n".join(pre))