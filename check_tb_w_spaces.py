from utils import map_ud_folders
import os
import pdb

mapper = map_ud_folders()

for uddir,tbname in mapper.items():
	for split in ["train","dev","covered-test"]:
		train_fn = os.path.join("2019","task2",uddir,tbname+"-um-"+split+".conllu")
		for line in open(train_fn,"r"):
			line = line.strip("\n")
			if line=="" or line[0]=="#":
				continue
			cols = line.split("\t")
			if " " in cols[1]:
				print(tbname," -- ",split)
				# print("--",cols[1],"--")
				# pdb.set_trace()
				break