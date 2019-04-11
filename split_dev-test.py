import sys, os
import numpy as np

infile = sys.argv[1]
pref = infile.rstrip(".conllu")

np.random.seed(42)
test_size = 100
sents = open(infile,'r').read().strip("\n").split("\n\n")
idxs = np.arange(len(sents))
np.random.shuffle(idxs)

with open(pref+"-dev.conllu",'w') as outdev:
	outdev.write("\n\n".join(sents[test_size:])+"\n")
	
with open(pref+"-test.conllu",'w') as outtest:
	outtest.write("\n\n".join(sents[:test_size])+"\n")

