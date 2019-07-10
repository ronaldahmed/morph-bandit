import sys,os
import numpy as np
from evaluate_2019_task2 import *
from pathlib import Path

exps = [
  "l1-a1",
  "l1-a2",
  "ml1-a1-bundle",
]
suff = "all"

if len(sys.argv)<2:
  print("usage: python eval_exps_sigm.py <mode> [<exp_id>]")
  sys.exit(1)

mode=sys.argv[1]
if len(sys.argv)>2:
  exps = [sys.argv[2]]
  suff = sys.argv[2]

tbnames = [x.strip('\n').split(" ") for x in open("data/tbnames-thesis","r") if x.strip("\n")!=""]
outfile = "experiments/%s-metrics.thesis" % suff

with open(outfile,'w') as outfile:
  print("tbname,exp,lem-acc,edist,m-acc,m-f1",file=outfile)
  for uddir,tb in tbnames:
    print(tb)
    gold_fn = "2019/task2/%s/%s-um-%s.conllu" % (uddir,tb,mode)

    for exp in exps:
      print("::",exp)
      pred_fn = "models_pred/%s-um-%s.conllu.%s.pred" % (tb,mode,exp)
      if not os.path.exists(pred_fn):
        continue
      reference = read_conllu(Path(gold_fn))
      output = read_conllu(Path(pred_fn))
      results = manipulate_data(input_pairs(reference, output))
      print(tb,exp,*["{0:.2f}".format(v) for v in results],sep=",",file=outfile)



