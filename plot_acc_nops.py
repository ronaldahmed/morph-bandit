import argparse
import os
import pandas as pd
import pdb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family' : 'serif',
        'size'   : 20}

matplotlib.rc('font', **font)


if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--gold", "-g", type=str, help="CSV with metrics")
  parser.add_argument("--pred", "-p", help="Load metrics",type=str)
  args = parser.parse_args()

  gold_data_to_plot = eval(open(args.gold,'r').read())
  pred_data_to_plot = eval(open(args.pred,'r').read())

  plt.figure()

  plt.scatter(pred_data_to_plot[-1],pred_data_to_plot[1],c=(1,0,0,0.6),marker="o",label="Predicted action sequences")

  # plt.scatter(gold_data_to_plot[-1],gold_data_to_plot[1],c=(0,0,1,0.8),marker=".")
  # plt.scatter(pred_data_to_plot[-1],pred_data_to_plot[1],c=(0,0.1,1,0.5),marker="s",label="Predicted action sequences")

  # for avg_gold,avg_pred,acc in zip(gold_data_to_plot[-1],\
  #                                  pred_data_to_plot[-1],\
  #                                  gold_data_to_plot[1]):
  #   plt.plot([avg_gold,avg_pred],[acc,acc],'b:')


  
  # plt.legend()
  plt.grid(True)
  plt.ylabel("Lemmata Accuracy")
  plt.xlabel("Average number of predicted actions")
  plt.show()


  # sns.distplot(gold_data_to_plot[-1], bins=50, kde=False, rug=False, label="Gold action sequences")
  # sns.distplot(pred_data_to_plot[-1], bins=50, kde=False, rug=False, label="Predicted action sequences")

  # nops = gold_data_to_plot[-1]
  # plt.figure()
  # plt.scatter(gold_data_to_plot[-1],gold_data_to_plot[1],c=(0,0,1,0.6),marker="o",label="Gold action sequences")
  # plt.scatter(pred_data_to_plot[-1],pred_data_to_plot[1],c=(1,0,0,0.6),marker="s",label="Predicted action sequences")
  
  print("------------>")