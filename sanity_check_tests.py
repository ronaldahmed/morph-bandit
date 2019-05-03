import argparse
from utils import *
import pdb

def form_lemma_sanity_check(tbname):
  print("Sanity check for lemmatizer oracle of operations...")
  print("TB: ",tbname)
  for split in ["train","dev","test"]:
    filename = "data/%s/%s" % (tbname,split)
    print("file:",filename)
    for line in open(filename,'r'):
      line = line.strip("\n")
      if line=="": continue
      form,lemma,feat,ops = line.split("\t")
      predicted = apply_operations(form,ops.split(),debug=True)

      if lemma.lower() != predicted:
        print("gold: %s | pred: %s | ops: %s" % (lemma,predicted,ops) )
        pdb.set_trace()


if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--tbname", "-t", type=str, help="Treebank name",required=True)
  args = parser.parse_args()

  form_lemma_sanity_check(args.tbname)