import argparse
from collections import defaultdict
from shared_task_reader import SharedTaskReader
from edist_bpe import EDistBPE
import os
from multiprocessing import Pool
from utils import map_ud_folders
import glob as gb
import pdb

def objective(tbname):
  print("::",tbname)
  reader = SharedTaskReader()
  train,dev,test = reader.read_task2(tbname)
  train_vocab = get_type_vocab(train)

  edbpe = EDistBPE(_num_merges=50,_inflector=False)

  if not os.path.exists("data/" + tbname):
    os.makedirs("data/" + tbname)
  # if args.mode == 'train':
  edbpe.train(train_vocab)
  edbpe.dump_merge_file(tbname)
  edbpe.encode(train,filename="data/%s/train" % tbname)
  edbpe.encode(dev,filename="data/%s/dev" % tbname)
  edbpe.encode(test,filename="data/%s/test" % tbname)
  return



def get_type_vocab(data):
  vocab = defaultdict(set)
  for sent in data:
    for form,lemma,_ in sent:
      vocab[lemma.lower()].add(form.lower())
  return vocab



if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--nj", "-j", type=int, help="Number of jobs", default=4)
  args = parser.parse_args()

  tbnames = list(map_ud_folders().values())

  with Pool(args.nj) as pool:
    pool.map(objective,tbnames)
