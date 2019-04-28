import argparse
from wfgen import WFGen

import pdb

if __name__=="__main__":

  parser = argparse.ArgumentParser() 
  parser.add_argument("--tbs", "-t", type=str, help="Treebank names, comma separated")
  parser.add_argument("--input", "-i", type=str, help="Input file to encode")
  parser.add_argument("--output", "-o", type=str, help="Outfile of encoded tokens")
  # parser.add_argument("--k", default=3,type=int, help="Number of folds")
  args = parser.parse_args()

  tbs = args.tbs.split(",")

  # read task 1 data
  wg = WFGen(_maxdata=150000,_num_merges=50)
  
  # ops = wg.get_primitive_actions("abc","at")
  # print(wg.encode_op_seq(ops))
  
  wg.read_task2(tbs)
  wg.dump_coded_sents_conllu(args.input,args.output)
  
