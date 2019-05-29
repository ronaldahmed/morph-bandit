import argparse
import os
import pandas as pd
import pdb
import matplotlib.pyplot as plt



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--gold", "-g", type=str, help="CSV with metrics")
  parser.add_argument("--pred", "-p", help="Load metrics",type=str)
  args = parser.parse_args()

  gold_data_to_plot = eval(open(args.gold,'r').read())
  pred_data_to_plot = eval(open(args.pred,'r').read())
  
  nops = gold_data_to_plot[-1]
  plt.figure()
  plt.scatter(nops,gold_data_to_plot[1],c=(1,0,0,0.8),marker="o",label="Gold sequences")
  plt.scatter(nops,gold_data_to_plot[1],c=(0,0,1,0.8),,marker="x",label="Predicted sequences")
  plt.legend()
  plt.grid("on")
  plt.ylabel("Lemmata Accuracy")
  plt.xlabel("Average number of actions")
  plt.show()

  
  print("------------>")