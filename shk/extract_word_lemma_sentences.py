import os
import glob as gb
import sys
sys.path.append("..")
import numpy as np
import xml.etree.ElementTree as ET
from utils_shk import *
from lex_utils import *
from word_type import WordType
from edist_bpe import EDistBPE
from utils import *
from collections import defaultdict

import pdb

basedir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")

def dump_conllu_crossval(data,fn):
  with open(fn,'w') as outfile:
    for sent in data:
      for i,(w,l,pos) in enumerate(sent):
        cols = ["_"]*10
        cols[0] = str(i+1)
        cols[1] = w
        cols[2] = l
        cols[3] = pos
        print(*cols,sep="\t",file=outfile)
      #
      print("",file=outfile)
  return


def run_editdist_bpe(train,test,fold):
  train_vocab = defaultdict(set)
  tbname = "shk_cuni"
  if not os.path.exists(os.path.join(basedir,tbname) ):
    os.makedirs(os.path.join(basedir,tbname))
  for sent in train:
    for w,l,_ in sent:
      train_vocab[w.lower()].add(l.lower())

  edbpe = EDistBPE(_num_merges=50,_inflector=False)
  edbpe.train(train_vocab)
  merge_file="merges."+str(fold)
  edbpe.dump_merge_file(tbname,merge_file,prefix=basedir)
  edbpe.encode(train,filename=os.path.join(basedir,tbname,"train.%d" % (fold)))
               # merge_file=os.path.join(basedir,tbname,merge_file))
  
  edbpe.encode(test,filename=os.path.join(basedir,tbname,"test.%d" % (fold)))
               # merge_file=os.path.join(basedir,tbname,merge_file))


if __name__ == "__main__":
  tagged_corpora_dir = 'corpus/corpusXml'
  
  vocab = set()
  affix_dict = get_affix_dict()

  dataset = []

  for filename in gb.glob(tagged_corpora_dir+"/*.xml"):
    tree = ET.ElementTree(file=filename)
    root = tree.getroot()
    for sentence in root:
      if sentence.tag == "lastSentence":
        continue #omit this cases
      firstWord = True
      
      sent_tuples = []

      for word in sentence:
        if not is_valid_annotation(word):
          continue
        candidate = WordType(word,affix_dict)
        lemma = candidate.lemma
        
        if lemma == None:
          sent_tuples = []
          print("-->empty lemma!!")
          pdb.set_trace()
          break

        sent_tuples.append((candidate.form,lemma,candidate.pos))
        vocab.add(candidate.pos)

      #END-WORDS
      if len(sent_tuples)>0:
        dataset.append(sent_tuples)
    #END-FOR-SENT
  #END-FOR-FN
  

  template = "../data/shk_cuni/shk_cuni-ud-%s-%d.conllu"
  num_folds = 10
  n_test = len(dataset) // num_folds
  idxs = np.arange(len(dataset))
  np.random.seed(42)

  for k in range(num_folds):
    test = [dataset[x] for x in idxs[:n_test]]
    train = [dataset[x] for x in idxs[n_test:]]

    dump_conllu_crossval(train,template % ("train",k))
    dump_conllu_crossval(test,template % ("test",k))

    run_editdist_bpe(train,test,k)

    np.random.shuffle(idxs)

