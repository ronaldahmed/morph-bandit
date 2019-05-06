import matplotlib
import matplotlib.pyplot as plt
import re
import pdb

sim_exps = [
	('de_gsd','en_ewt'),
	('it_isdt','es_ancora'),
	('fr_gsd','es_ancora'),
]

dist_exps = [
	('ar_padt','en_ewt'),
	('es_ancora','en_ewt'),
]


colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def plot_exp(exps,title):
	plt.figure(figsize=(12,12))
	plt.suptitle(title)

	for exp,color in zip(exps,colors[:len(exps)]):
		exp_name = exp[0] + "-" + exp[1]
		filename = "emb/multi/%s/%s/train.log" % (exp_name,exp_name)
		logs = []
		for line in open(filename,'r'):
			line = line.strip("\n")
			if line=="": continue
			if "__log__" in line:
				idx = line.find("__log__")

				log = eval(line[idx+8:])
				logs.append(log)
		#
		iters = [int(x["n_iter"]) for x in logs]
		nw = 1
		w_prf = 330
		keys = list([x for x in logs[0].keys() if x != "n_iter"])
		keys.sort()

		for i,field in enumerate(keys):
			vals = [x[field] for x in logs]
			plt.subplot(w_prf+nw)
			plt.plot(iters,vals,color=color,label=exp_name)
			plt.title(field)
			nw += 1
		#
	#
	plt.legend()
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()
	plt.show()


###
#plot_exp(sim_exps,"Similar languages")
plot_exp(dist_exps,"Distant languages")