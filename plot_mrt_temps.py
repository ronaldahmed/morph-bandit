import numpy as np
import os
import re
import pickle
import pdb


def saveObject(obj, name='model'):
  with open(name + '.pickle', 'wb') as fd:
    pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
  # Load tagger
  with open(obj_name, 'rb') as fd:
    obj = pickle.load(fd)
  return obj



server_mode = True


bs = 5
alpha = 1e-3
temp = 1
temps = ["1", "5", "10"]

root = os.path.join("models-segm","es_ancora")
folder_name_template = "l1.mrt.warm_optm-adadelta_alpha-%s_sample-%s_clip-0_bs-%d"

res_pat = re.compile(r'\s+dev.+acc:\s+(?P<acc>[.0-9]+)[%],\s+dist:\s+(?P<dist>[.0-9]+)')

acc_d = {}
edist_d = {}

if server_mode:

	folders =[
		"l1.mrt.warm_optm-adadelta_alpha-0.0001_sample-20_clip-0_bs-5",
		"l1.mrt.warm_optm-adadelta_alpha-0.0001_sample-20_clip-0_bs-5_temp-5",
		"l1.mrt_optm-adadelta_alpha-0.0001_sample-20_clip-0_bs-5_temp-10",
	]

	for t,foldername in zip(temps,folders):
		acc_d[t] = []
		edist_d[t] = []

		fname = os.path.join(root,foldername,"log.out")

		for line in open(fname,'r'):
			line = line.strip("\n")
			if line=='': continue
			match = res_pat.search(line)
			if match is None: continue
			acc = float(match.group("acc"))
			edist = float(match.group("dist"))
			acc_d[t].append(acc)
			edist_d[t].append(edist)
	#
	saveObject([acc_d,edist_d],"mrt_temps")

else:
	acc_d,edist_d = uploadObject("mrt_temps.pickle")

	# pdb.set_trace()

	import matplotlib
	import matplotlib.pyplot as plt

	font = {'family' : 'serif',
	        'size'   : 20 }

	matplotlib.rc('font', **font)


	eps = np.arange(len(acc_d[ssizes[0]]))
	colors_d = dict(zip(ssizes,['r','b','g']))
	line_d = dict(zip(ssizes,['--','-','-.']))

	plt.figure(figsize=(16,8))
	plt.subplot(121)
	for t in temps:
		_len = min(7,len(acc_d[t]))
		plt.plot(eps[:_len],acc_d[t][:_len],colors_d[t]+line_d[t],label=r"$\tau$="+t)
	plt.grid(True)
	plt.xlabel("Epoch")
	plt.ylabel("Lemmata Accuracy")

	plt.subplot(122)
	for t in temps:
		_len = min(7,len(edist_d[t]))
		plt.plot(eps[:_len],edist_d[t][:_len],colors_d[t]+line_d[t],label=r"$tau$="+t)
	plt.grid(True)
	plt.xlabel("Epoch")
	plt.ylabel("Levenshtein Distance")

	plt.legend(loc='upper right',bbox_to_anchor=(1.5, 1))

	plt.tight_layout()

	plt.show()