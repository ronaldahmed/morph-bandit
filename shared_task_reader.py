from collections import defaultdict
from label_dictionary import LabelDictionary
from utils import *
import os


class SharedTaskReader:
  def __init__(self,_maxdata=1000000):
    self.max_data = _maxdata
    # ud mappers
    self.uddir2tbname = map_ud_folders()
    self.tbname2uddir = {v:k for k,v in self.uddir2tbname.items()}
    # label dicts
    self.form_vocab = LabelDictionary()
    self.lemma_vocab = LabelDictionary()
    self.feat_vocab = LabelDictionary()


  def read_accum_by_lang_task1(self,langs):
    """ read from several treebanks and accumulate by language"""
    count = 0
    for lid in langs:
      lang = iso_mapper[lid]
      dirname = gb.glob("2019/task1/"+lang+"--*")[0]
      filename = os.path.join(dirname,lang+"-train-high")
      for line in open(filename,'r'):
        line = line.strip("\n")
        if line=='': continue
        lem,form,labels = line.split("\t")
        #self.data[lid].append([lem,form,])

        actions = self.get_primitive_actions(lem,form)
        self.op_seqs.append(actions)
        self.label_lid.append([labels.split(";"),lid])

        if count % 1000 == 0:
          print("->",count)
        count += 1

        if count > self.max_data:
          break
    #


  def read_task1(self,src_tgt_code):
    """ read from several treebanks and accumulate by language"""
    count = 0
    src_lang,tgt_lang = src_tgt_code.split("--")
    src_train = tgt_train = tgt_dev = tgt_test = []
    for data,filename in zip([src_train, tgt_train, tgt_dev, tgt_test],
                              [src_lang+"-train-high",
                               tgt_lang+"-train-low",
                               tgt_lang+"-dev",
                               tgt_lang+"-test-covered"]):
      filename = os.path.join("2019/task1/",filename)

      for line in open(filename,'r'):
        line = line.strip("\n")
        if line=='': continue
        lem,form,labels = line.split("\t")
        data.append([form,lem,labels])
        
        if count % 1000 == 0:
          print("->",count)
        count += 1

        if count > self.max_data:
          break
    #
    return 


  def read_conllu(self,filename):
    data = []
    sent  = []
    for line in open(filename,'r'):
      line = line.strip("\n")
      if line.startswith("#"):
        continue
      if line=="":
        data.append(sent)
        sent = []
      else:
        cols = line.split("\t")
        lem,form = cols[2], cols[1]
        # lem = lem.lower().replace(" ",SPACE_LABEL)
        # form = form.lower().replace(" ",SPACE_LABEL)
        feats = cols[5]
        sent.append([form,lem,feats])
    #
    return data

  def read_task2(self,tbname=None,uddir=None):
    count = 0

    if uddir!=None and tbname==None:
      tbname = self.uddir2tbname[uddir]
    if uddir==None and tbname!=None:
      uddir = self.tbname2uddir[tbname]

    train = self.read_conllu("2019/task2/%s/%s-um-train.conllu" % (uddir,tbname))
    dev   = self.read_conllu("2019/task2/%s/%s-um-dev.conllu" % (uddir,tbname))
    test  = self.read_conllu("2019/task2/%s/%s-um-covered-test.conllu" % (uddir,tbname))

    return train,dev,test
