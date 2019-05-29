import argparse
import os
import pandas as pd
import pdb
import matplotlib.pyplot as plt


def get_gold_ops(filename):
  accum, count = 0.0,0.0
  for line in open(filename,"r"):
    line = line.strip("\n")
    if line=="":  continue
    comps = line.split("\t")
    if len(comps)==4:
      accum += len(comps[3].split(" ")) - 2
    else:
      accum += len(comps) - 2 - 3
    count += 1
  return accum / count


def get_pred_ops(filename):
  accum, count = 0.0,0.0
  for line in open(filename,"r"):
    line = line.strip("\n")
    if line=="":  continue
    comps = line.split("\t")
    accum += len([x for x in comps[9].split(" ") if x!=""])
    count += 1
  return accum / count



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--results", "-r", type=str, help="CSV with metrics")
  
  parser.add_argument("--use_gold", "-u", help="Use gold or pred",action="store_true")
  args = parser.parse_args()

  results = pd.read_csv("experiments/res-anlz.dev.csv")
  results.index = results["treebank"]

  filename = args.load

  data_to_plot = [] # [tbnames, lem_acc, edist,avg_nops]
  
  use_gold = args.use_gold
  filename = "dev" if use_gold else "dev.anlz.conllu.pred"

  for root,dirnames,_ in os.walk("data/"):
    for tbname in dirnames:
      if tbname=="ru_syntagrus":
        continue
      fn = os.path.join("data",tbname,filename)
      avg = 0
      if use_gold:
        avg = get_gold_ops(fn)
      else:
        avg = get_pred_ops(fn)
      data_to_plot.append([tbname,results["lem_acc"][tbname],results["edist"][tbname],avg])
  
  data_to_plot = list(zip(*data_to_plot))
  suff = "gold" if use_gold else "pred"
  with  open("experiments/acc_nops."+suff,'w') as outfile:
    print(data_to_plot,file=outfile)
  
  