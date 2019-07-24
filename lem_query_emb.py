import sys
import torch
import numpy as np
from time import monotonic
from my_flags import *
from data_utils import *
from model_analizer import Analizer
from model_lemmatizer import Lemmatizer
from trainer_analizer import TrainerAnalizer
from trainer_lemmatizer import TrainerLemmatizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from utils import STOP_LABEL, SPACE_LABEL, apply_operations

from gensim.models import Word2Vec


def get_vocab_from_vec(tbname):
  fn = "../thesis-files/l1-mono-emb/"+tbname+".vec"
  words = [ss.split()[0] for ss in open(fn,'r') if ss.strip("\n")!=""]
  return words[1:]

def dump_multi_vec(wforms,tbname,outfn):
  lid = tbname[:2]
  outfile = open(outfn,'w')
  mfn = "../thesis-files/l1-multi-emb/%s-es/%s-es/vectors-%s.pth" % (lid,lid,lid)
  emb_mtx = torch.load(mfn,map_location='cpu')["vectors"].contiguous().cpu().numpy()
  assert emb_mtx.shape[0] == len(wforms)
  print(*emb_mtx.shape,sep=" ", file=outfile)
  for idx,form in enumerate(wforms):
    str_emb = " ".join([str(x) for x in emb_mtx[idx,:]])
    print(form,str_emb,sep=" ", file=outfile)
  return



if __name__ == '__main__':
  prepro = True
  tbnames = [
    "es_ancora",
    "cs_pdt",
    "en_ewt",
  ]

  if prepro:
    for tb in tbnames:
      print("::",tb)
      outfn = "../thesis-files/l1-multi-emb/"+tb+".vec"
      wforms = get_vocab_from_vec(tb)
      dump_multi_vec(wforms,tb,outfn)

    sys.exit(0)
  else:
    args = analizer_args()
    print(args)
    w2vmodel = {}
    for tb in tbnames:
      print("::",tb)
      infn = "../thesis-files/l1-multi-emb/"+tb+".vec"
      model = Word2Vec.load_word2vec_format(infn)
      w2vmodel[tb] = model
    #





  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

  input_temp = "data/%s/%s"
  emb_temp = "../thesis-files/l1-multi-emb/%s-es/%s-es/vectors-%s.pth"


  for tb in tbnames:
    exp_args = args
    exp_args["train_file"] = input_temp % (tb,"train")
    lid = tb[:2]
    emb_fn = emb_temp % (lid,lid,lid)

    loader = DataLoaderAnalizer(exp_args)
    train = loader.load_data("train")


