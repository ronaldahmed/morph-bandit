import os
import sys
import glob as gb
import numpy as np
import pdb

tbnames = open("data/top-bot-n","r").read().strip("\n").strip(" ").split("\n")
tbnames = [x.split(" ")[1] for x in tbnames]

outfile = open("experiments/top-bot-n-multi.csv","w")

metrics = {}

print("tbname,lem_lacc,lem_edist,anlz_lacc,anlz_edist,m_acc,m_f1",file=outfile)

for tbname in tbnames:
	metrics[tbname] = [[]]*6
	for fn in gb.glob("models-ens/"+tbname+"/lem-*-log.out"):
		line = open(fn,"r").read().strip("\n").strip(" ").split("\n")[-1]
		_,lacc,ed = line.split("\t")
		lacc = float(lacc)
		ed = float(ed)
		metrics[tbname][0].append(lacc)
		metrics[tbname][1].append(ed)
	for fn in gb.glob("models-ens/"+tbname+"/anlz-*-log.out"):
		line = open(fn,"r").read().strip("\n").strip(" ").split("\n")[-1]
		res = line.split("\t")
		res = [float(x) for x in res[1:]]
		for i in range(1,4):
			metrics[tbname][i+1].append(res[i])
	#
	pdb.set_trace()	
	print("%s,%s" % (tbname,",".join(["%.2f(%.2f)"%(np.mean(x),np.std(x)) for x in metrics[tbname]]) ), file=outfile)

