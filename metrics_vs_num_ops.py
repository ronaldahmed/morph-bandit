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
    accum += len(comps[9].split(" ")) - 2
    count += 1
  return accum / count



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--results", "-r", type=str, help="CSV with metrics")
  parser.add_argument("--load", "-l", help="Load metrics",action="store_true")
  args = parser.parse_args()

  results = pd.read_csv("res-anlz.dev.csv")
  results.index = results["treebank"]

  use_gold = True
  filename = "dev" if use_gold else "dev.anlz.conllu.pred"

  data_to_plot = [] # [tbnames, lem_acc, edist,avg_nops]
  
  if not args.load:

    for root,dirnames,_ in os.walk("data/"):
      for tbname in dirnames:
        fn = os.path.join("data",tbname,filename)
        avg = 0
        if use_gold:
          avg = get_gold_ops(fn)
        else:
          avg = get_pred_ops(fn)
        data_to_plot.append([tbname,results["lem_acc"][tbname],results["edist"][tbname],avg])
    
    data_to_plot = list(zip(*data_to_plot))

    with  open("metrics_vs_nops.eval",'w') as outfile:
      print(data_to_plot,file=outfile)
  else:
    data_to_plot = eval(open("metrics_vs_nops.eval",'r').read())

  # nops = data_to_plot[-1]
  # plt.figure()
  # plt.scatter(nops,data_to_plot[1],c="red",marker="o")
  # plt.scatter(nops,data_to_plot[2],c="blue",marker="s")

  # plt.show()

  
  print("------------>")