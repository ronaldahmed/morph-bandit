import sys
from sklearn.metrics import f1_score
from utils import uploadObject
import numpy as np

import pdb

gold_fn = sys.argv[1]
pred_fn = sys.argv[2]
feats_vocab_fn = sys.argv[3]


glines = open(gold_fn,'r').read().strip("\n").split("\n")
plines = open(pred_fn,'r').read().strip("\n").split("\n")
feats_vocab = uploadObject(feats_vocab_fn)

gold_feats = []
pred_feats = []
acc = 0.0
acc_bundle = 0.0

for gline,pline in zip(glines,plines):
	gline = gline.strip("\n")
	pline = pline.strip("\n")
	if gline.startswith("#"): continue
	if gline=="": continue
	gcols = gline.split("\t")
	pcols = pline.split("\t")
	gfeats = gcols[5].split(";")
	pfeats = pcols[5].split(";")
	gfeats.sort()
	pfeats.sort()
	acc += int(";".join(gfeats)==";".join(pfeats))
	acc_bundle += int(gcols[5]==pcols[5])
	gold_feats.append([feats_vocab.get_label_id(x) for x in gfeats])
	pred_feats.append([feats_vocab.get_label_id(x) for x in pfeats])
	
	if ";".join(gfeats) != gcols[5]:
		print("%20s %20s | %20s %20s" % (gcols[5],pcols[5],";".join(gfeats),";".join(pfeats)) )
	# pdb.set_trace()
#


nw = len(gold_feats)
gmat = np.zeros([nw,len(feats_vocab)])
pmat = np.zeros([nw,len(feats_vocab)])
for i in range(nw):
	gmat[i,gold_feats[i]] = 1.0
	pmat[i,pred_feats[i]] = 1.0


msd_acc = (100.0*acc) / nw
msd_acc_bundle = (100.0*acc_bundle) / nw
f1 = 100.0*f1_score(gmat,pmat,average="micro") # average is pessimistic

print("%.2f %.2f %.2f" % (msd_acc_bundle,msd_acc,f1))