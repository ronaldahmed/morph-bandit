import argparse
from collections import defaultdict
from shared_task_reader import SharedTaskReader
from edist_bpe import EDistBPE
import os

import pdb

def get_type_vocab(data):
  vocab = defaultdict(set)
  for sent in data:
    for form,lemma,_ in sent:
      vocab[lemma].add(form)
  return vocab


if __name__=="__main__":

  parser = argparse.ArgumentParser() 
  parser.add_argument("--tb", "-t", type=str, help="Treebank name")
  parser.add_argument("--input", "-i", type=str, help="Conllu input file to encode")
  parser.add_argument("--mode", "-m", type=str, help="Mode [train,encode]")
  parser.add_argument("--vocab", "-v", type=str, help="Merge history filename")
  args = parser.parse_args()

  reader = SharedTaskReader()
  train,dev,test = reader.read_task2(tbname=args.tb)
  train_vocab = get_type_vocab(train)

  edbpe = EDistBPE(_num_merges=50,_inflector=False)

  if not os.path.exists("data/" + args.tb):
    os.makedirs("data/" + args.tb)
  # if args.mode == 'train':
  print("Training BPE merges...")
  edbpe.train(train_vocab)
  edbpe.dump_merge_file(args.tb)
  print("Encoding training data...")
  edbpe.encode(train,filename="data/%s/train" % args.tb)
  print("Encoding dev data...")
  edbpe.encode(dev,filename="data/%s/dev" % args.tb)
  print("Encoding test data...")
  edbpe.encode(test,filename="data/%s/test" % args.tb)
