import argparse
import os
import pandas as pd
import pdb
import matplotlib.pyplot as plt



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--gold", "-r", type=str, help="CSV with metrics")
  parser.add_argument("--pred", "-l", help="Load metrics",type=str)
  args = parser.parse_args()

  gold_data_to_plot = eval(open(args.gold,'r').read())
  pred_data_to_plot = eval(open(args.pred,'r').read())
  
  nops = gold_data_to_plot[-1]
  plt.figure()
  plt.scatter(nops,data_to_plot[1],c="red",marker="o")
  #plt.scatter(nops,data_to_plot[2],c="blue",marker="s")
  plt.grid("on")
  plt.show()

  
  print("------------>")