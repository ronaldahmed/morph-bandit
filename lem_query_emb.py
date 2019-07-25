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


def get_neighbors(a_emb,model,ops_by_tuple,nmorphs=10,nstarts=10):
  def get_ex(actions):
    res = []
    ex_thr = 5
    for m,_ in actions:
      cnt = 0
      res.append([])
      for (w,l),ops in ops_by_tuple.items():
        if m in ops:
          res[-1].append(w+","+l)
          cnt += 1
        if cnt>ex_thr-1: break
    return res

  cands = model.most_similar(positive=[a_emb],negative=[],topn=300)
  morphs = [x for x in cands if not x[0].startswith("START")][:nmorphs]
  starts = [x for x in cands if x[0].startswith("START")][:nstarts]
  ex_morphs = get_ex(morphs)
  ex_starts = get_ex(starts)

  return starts,morphs,ex_starts,ex_morphs


def print_res(lid,cand):
  print("\t",lid)
  ss,mm,es,em = cand
  for (l,s),ex in zip(mm,em):
    print("\t\t%20s (%.2f) | %s" % (l,s,"  ".join(ex)) )

  for (l,s),ex in zip(ss,es):
    print("\t\t%20s (%.2f) | %s" % (l,s,"  ".join(ex)) )


def load_train_tuples(tb):
  filename = "data/"+tb+"/train"
  tups = {}
  for line in open(filename,'r'):
    line = line.strip('\n')
    if line=="": continue
    comps = line.split("\t")
    w,lem,feats = comps[:3]
    op_seq = comps[3:]
    _key = tuple([w,lem])
    if _key not in tups:
      tups[_key] = op_seq
  return tups

##################################################################################3

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
    op_by_tuple = {}
    for tb in tbnames:
      print("::",tb)
      infn = "../thesis-files/l1-multi-emb/"+tb+".vec"
      model = KeyedVectors.load_word2vec_format(infn)
      w2vmodel[tb] = model
      op_by_tuple[tb] = load_train_tuples(tb)
    #

    queries = [
      ("A_-s","es_ancora"),  # Pl
      ("A_-y","cs_pdt"),  # Pl
      ("A_-ía","es_ancora"), # PST
      ("A_-ed","en_ewt"), # PST
      ("_A-i","es_ancora"),  # Neg
      ("_A-in","es_ancora"),  # Neg
      ("_A-im","es_ancora"),  # Neg
      ("_A-dis","es_ancora"),  # Neg
      ("A_-ne","cs_pdt"),  # Neg
      # get_action_queries(w2vmodel["es_ancora"].vocab.keys(),"A_-ando"),
      # get_action_queries(w2vmodel["es_ancora"].vocab.keys(),"A_-ndo"),
    ]
    for qry_pat,tb in queries:
      src_model = w2vmodel[tb]
      for action in get_action_queries(src_model.vocab.keys(),qry_pat):
        emb = src_model[action]
        cs_nb = get_neighbors(emb,w2vmodel["cs_pdt"],op_by_tuple["cs_pdt"])
        en_nb = get_neighbors(emb,w2vmodel["en_ewt"],op_by_tuple["en_ewt"])
        es_nb = get_neighbors(emb,w2vmodel["es_ancora"],op_by_tuple["es_ancora"])

        print(":: ",tb,"--",action)
        for lid,cand in zip(["es","en","cs"],[es_nb,en_nb,cs_nb]):
          print_res(lid,cand)

        # pdb.set_trace()


    print("-->")