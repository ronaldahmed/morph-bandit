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

from gensim.models import Word2Vec, KeyedVectors

import pdb

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


def get_action_queries(vocab,ending):
  res = []
  for w in vocab:
    if w.endswith(ending):
      res.append(w)
  return res


def get_neighbors(a_emb,model,top=20):
  return model.most_similar(positive=[a_emb],negative=[],topn=top)


def print_res(lid,cand):
  print("\t",lid)
  for l,s in cand:
    print("\t\t%20s\t%.4f" % (l,s) )


if __name__ == '__main__':
  prepro = False
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
    # args = analizer_args()
    # print(args)
    w2vmodel = {}
    for tb in tbnames:
      print("::",tb)
      infn = "../thesis-files/l1-multi-emb/"+tb+".vec"
      model = KeyedVectors.load_word2vec_format(infn)
      w2vmodel[tb] = model
    #

    src_model = w2vmodel["es_ancora"]
    queries = [
      get_action_queries(src_model.vocab.keys(),"A_-s"),  # Pl
      get_action_queries(src_model.vocab.keys(),"A_-ía"), # PST
      get_action_queries(src_model.vocab.keys(),"_A-i"),  # Neg
      # get_action_queries(w2vmodel["es_ancora"].vocab.keys(),"A_-ando"),
      # get_action_queries(w2vmodel["es_ancora"].vocab.keys(),"A_-ndo"),
    ]
    for qry in queries:
      for action in qry:
        emb = src_model[action]
        cs_nb = get_neighbors(emb,w2vmodel["cs_pdt"])
        en_nb = get_neighbors(emb,w2vmodel["en_ewt"])
        es_nb = get_neighbors(emb,w2vmodel["es_ancora"])

        print(":: ",action)
        for lid,cand in zip(["es","en","cs"],[es_nb,en_nb,cs_nb]):
          print_res(lid,cand)

        pdb.set_trace()


    print("-->")