import numpy as np
import os
import re
import pickle



def saveObject(obj, name='model'):
  with open(name + '.pickle', 'wb') as fd:
    pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
  # Load tagger
  with open(obj_name, 'rb') as fd:
    obj = pickle.load(fd)
  return obj



server_mode = True


bs = 10
ss = 20
temp = 1
alphas = [0.05, 0.001, 0.0001, 0.00001]

root = os.path.join("..","models-segm","es_ancora")
folder_name_template = "l1.mrt.warm_optm-adadelta_alpha-%f_sample-%d_clip-0_bs-%d"

res_pat = re.compile(r'\s+dev.+acc:\s+(?P<acc>[.0-9]+[%]),\s+dist:\s+(?P<dist>[.0-9]+)')

acc_d = {}
edist_d {}

if server_mode:

	for a in alphas:
		acc_d[a] = []
		edist_d[a] = []

		foldername = folder_name_template % (a,ss,bs)
		fname = os.path.join(foldername,"log.out")
		for line in open(fname,'r'):
			line = line.strip("\n")
			if line=='': continue
			match = res_pat.search(line)
			if match is None: continue
			acc = float(match.groups("acc"))
			edist = float(match.groups("dist"))
			acc_d[a].append(acc)
			edist_d[a].append(edist)
	#
	saveObject([acc_d,edist_d],"mrt_alphas")

else:
	acc_d,edist_d = uploadObject("mrt_alphas.pickle")

	import matplotlib
	import matplotlib.pyplot as plt

	font = {'family' : 'serif',
	        'size'   : 20 }

	matplotlib.rc('font', **font)


	eps = np.arange(len(acc_d[alphas[0]]))
	colors_d = dict(zip(alphas,['c','g','b','r']))

	plt.figure(figsize=(12,8))
	plt.subplot(121)
	for a in alphas:
		plt.plot(eps,acc_d[a],colors_d[a]+"-",label=r"$\alpha$="+str(a))
	plt.grid(True)

	plt.subplot(122)
	for a in alphas:
		plt.plot(eps,edist_d[a],colors_d[a]+"-",label=r"$\alpha$="+str(a))
	plt.grid(True)