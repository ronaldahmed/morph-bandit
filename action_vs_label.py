import argparse
from wfgen import WFGen

import pdb

if __name__=="__main__":

  parser = argparse.ArgumentParser() 
  parser.add_argument("--langs", "-l", type=str, help="Lang iso ids, comma separated")
  # parser.add_argument("--k", default=3,type=int, help="Number of folds")
  args = parser.parse_args()

  langs = args.langs.split(",")

  # read task 1 data
  wg = WFGen(_maxdata=150000,_num_merges=50)
  wg.read_task1(langs)
  wg.oracle_bpe()
  #wg.plot_corr_labels_actions()
  wg.plot_corr_bpe_ops(langs[0])
