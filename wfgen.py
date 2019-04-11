import os, sys
import glob as gb
import numpy as np
import pdb
import pickle

from collections import defaultdict
from label_dictionary import LabelDictionary
from utils import *

from op_typeins import *

import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt



#########################################################################################

class WFGen:
  def __init__(self,_maxdata=2000,_num_merges=10,_inflector=False):
    self.label_dict = {}
    self.label_action = defaultdict(int)
    self.lang_action = defaultdict(int)
    # direction of analysis
    #  lem->form : inflector True
    #  form->lem : inflector False
    self.inflector = _inflector

    # ud mappers
    self.uddir2tbname = map_ud_folders()
    self.tbname2uddir = {v:k for k,v in self.uddir2tbname.items()}

    # operation dictionary
    self.op_dict = LabelDictionary()
    self.populate_op_ids()

    # keep track of data for counts
    self.op_seqs = []
    self.label_lid = []
    # bigram operation counts
    
    # number of BPE merges / ops
    self.num_merges = _num_merges
    self.merge_history = []
    self.max_data = _maxdata
    self.types_data = {}


  def populate_op_ids(self):
    # order here coincides with EditDist scoring sequence
    for _op in [DEL,INS,SUBS,TRSP,SKIP]:
      self.op_dict.add(_op)


  def read_task1(self,langs):
    count = 0
    for lid in langs:
      lang = iso_mapper[lid]
      dirname = gb.glob("task1/"+lang+"--*")[0]
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


  def read_task2(self,tb_names):
    count = 0
    for tbname in tb_names:
      uddir = self.tbname2uddir[tbname]
      filename = "task2/%s/%s-um-train.conllu" % (uddir,tbname)

      # collect per-lang vocab
      vocab = {} # { lem::form : [FEAT_SET.1,FEAT_SET.2,...] }
      for line in open(filename,'r'):
        line = line.strip("\n")
        if line=="" or line.startswith("#"):
          continue
        cols = line.split("\t")
        lem_form = cols[2].lower() + "::" + cols[1].lower()
        tmp = cols[5].split(";")
        tmp.sort()
        feats = ";".join(tmp)
        if lem_form not in vocab:
          vocab[lem_form] = set()
        vocab[lem_form].add(feats)
      #

      # get action sequence
      lem_form_list = []
      for lem_form,feat_set in vocab.items():
        if lem_form=="::::":
          lem = form = ":"
        else:
          lem,form = lem_form.split("::")
        
        actions = self.get_primitive_actions(lem,form)
        self.op_seqs.append(actions)

        #self.types_data.append(TypeInstance(lem,form))
        lem_form_list.append([lem,form])
        
        if count % 1000 == 0:
          print("->",count)
        count += 1

        if count > self.max_data:
          break
    #
    self.oracle_bpe()
    for lem_form,ops in zip(lem_form_list,self.op_seqs):
      self.types_data[tuple(lem_form)] = ops



  def dump_coded_sents_conllu(self,filename,outfn):
    count = 0
    sent = []
    outfile = open(outfn,"w")

    for line in open(filename,'r'):
      line = line.strip("\n")
      if line.startswith("#"):
        continue
      if line=="":
        print(" ".join(sent),file=outfile)
        sent = []

      else:
        cols = line.split("\t")
        lem,form = cols[2].lower(), cols[1].lower()
        ops = []
        if (lem,form) in self.types_data:
          ops = self.types_data[(lem,form)]
        else:
          ops = self.get_primitive_actions(lem,form)
        op_tokens = self.encode_op_seq(ops)
        orig = lem if self.inflector else form
        orig_tok = "%s.%s-%s" % (START,START,orig)
        sent.append(" ".join([orig_tok] + op_tokens))

        # print(op_tokens)
        # if count % 5 == 0:
        #   pdb.set_trace()
        #   print("---")
        count += 1


  def get_primitive_actions(self,str1,str2):
    """ algorithm based on DP edit-distance (Damerau–Levenshtein distance)  """
    # 1: lemma
    # 2: form
    # print(lemma,"--",form)
    if not self.inflector:
      str2,str1 = str1,str2

    n = len(str1)
    m = len(str2)
    nm_max = max(n,m)
    dp = MAX_INT*np.ones([nm_max+1,nm_max+1])
    for i in range(nm_max+1):
      dp[i,0] = i
      dp[0,i] = i
    
    bp = {}
    for i in range(nm_max+1):
      bp[(i,0)] = [-1,-1,'']
      bp[(0,i)] = [-1,-1,'']

    for i in range(1,n+1):
      for j in range(1,m+1):
        if i-1<m and j-1<n and str2[i-1] == str1[j-1]:
          dp[i,j] = dp[i-1,j-1]
          bp[i,j] = [i-1,j-1,SKIP]
        else:
          # transposition op conds
          scores = np.array([dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+1,MAX_INT])

          if i-1<m and j-1<n and i>1 and j>1 and \
             str1[j-1]==str2[i-2] and str1[j-2]==str2[i-1]:
            scores[-1] = dp[i-2,j-2] + 1

          dp[i,j] = scores.min()
          action = self.op_dict.get_label_name(scores.argmin())
          if action==DEL:
            bp[i,j] = [i-1,j]
          elif action==INS:
            bp[i,j] = [i,j-1]
          elif action==TRSP:
            bp[i,j] = [i-2,j-2]
          else:
            bp[i,j] = [i-1,j-1]
          bp[i,j].append(action)
        #
      #
    #

    # print("best score:",dp[n,m])
    u = bp[n,m]
    actions = []
    while u != [-1,-1,'']:
      if u[2] != SKIP:
        op = Operation(u[2],u[0],str1,True)
        actions.append(op)
      u = bp[u[0],u[1]]

    #for a in reversed(actions):
    #  print(a)
    actions.sort(key=lambda x: (x.pos,x.name))
    actions.append(Operation(STOP,n,str1,True))

    #pdb.set_trace()

    return actions


  def encode_op_seq(self,ops):
    # op_name_mask* - segment
    ops = ops[:-1] # all but STOP
    n_ops = len(ops)
    prefs_cnt = -1
    sufs_cnt = n_ops+1
    last_idx = 0
    for i,op in enumerate(ops):
      # test for prefix
      if op.pos != last_idx:
        break
      prefs_cnt = i
      last_idx = op.pos + len(op.segment)
    last_idx = n_ops
    for i in range(n_ops-1,-1,-1):
      op = ops[i]
      if op.pos != last_idx - len(op.segment):
        break
      sufs_cnt = i

    if prefs_cnt > sufs_cnt:
      print("prefs > sufs!!")
      pdb.set_trace()

    # encoding
    sent = []
    for i,op in enumerate(ops):
      op_mask_code = "A_B"
      if i <= prefs_cnt:
        op_mask_code = "_A"
      if i >= sufs_cnt:
        op_mask_code = "A_"
      token = "%s.%s-%s" % (op.name,op_mask_code,op.segment)
      sent.append(token)
    sent.append(STOP+"."+STOP+"-</>")

    return sent


  ##################################################################################

  def clean_plot_vars(self,):
    self.label_action = defaultdict(int)
    self.lang_action = defaultdict(int)


  def accum_lab_act_counts(self,actions,labels,lang):
    for lab in labels:
      for act in actions:
        self.label_action[(lab,act)] += 1
        self.lang_action[(lang,act)] += 1
    #


  def plot_cooc(self,_dict,title,annot=True):
    dict_no_skip = {x:y for x,y in _dict.items() if x[1]!="skip"}

    df = pd.Series(list(_dict.values()),index=pd.MultiIndex.from_tuples(_dict.keys()))
    out = df.unstack().fillna(0)
    print(out)

    fig = plt.figure(figsize=(12,8))
    
    sb.heatmap(out,cmap="YlGnBu",annot=annot)
    plt.title(title)
    plt.tight_layout()

    # plt.subplot(122)
    # df = pd.Series(list(dict_no_skip.values()),index=pd.MultiIndex.from_tuples(dict_no_skip.keys()))
    # out = df.unstack().fillna(0)
    # sb.heatmap(out,cmap="YlGnBu",annot=True)

    # fig.suptitle(title)


  def plot_corr_labels_actions(self):
    for seq,labs,lid in zip([self.op_seqs,self.label_lid[0],self.label_lid[1]]):
      act_names = [a.name for a in actions]
      self.accum_lab_act_counts(act_names,labs,lid)

    self.plot_cooc(self.label_action,"actions vs labels")
    self.plot_cooc(self.lang_action,"actions vs languages")
    plt.show()


  def plot_corr_bpe_ops(self,lang="en"):
    self.clean_plot_vars()
    for seq,labs_lid in zip(self.op_seqs,self.label_lid):
      act_names = ["%s-%s" % (a.name,a.segment) for a in seq if a.name != STOP]
      self.accum_lab_act_counts(act_names,labs_lid[0],labs_lid[1])

    self.plot_cooc(self.label_action,lang + " - prim.ops vs labels",False)
    plt.show()



  #################################################################################
  def join_operation(self,op1,op2):
    jname = op1.name if op1.name == op2.name else op1.name+"_"+op2.name
    new_op = Operation(jname,min(op1.pos,op2.pos),op1.form,False)
    new_op.update_mask(op1.mask | op2.mask)
    return new_op


  def get_affix_stats(self,op_vocab):
    pairs = defaultdict(int)
    for seq in op_vocab:
      if len(seq) < 3: continue # don't count stop op
      for i in range(len(seq)-2):
        # only count if adjacent and same type of OP
        op1 = seq[i]
        op2 = seq[i+1]
        if op1.name != op2.name or op1.pos + len(op1.segment) != op2.pos:
          continue
        pairs[op1,op2] += 1
    return pairs


  def merge_affix_oracle(self,best,v_in):
    v_out = []
    op1,op2 = best
    count = 0
    for seq in v_in:
      new_seq = []
      i = 0
      while i < len(seq):
        if i<len(seq)-1 and op1==seq[i] and op2==seq[i+1]:
          new_seq.append( self.join_operation(seq[i],seq[i+1]) )
          i += 1
        else:
          new_seq.append(seq[i])
        i += 1
      #
      v_out.append(new_seq)
      if count % 1000 == 0:
        print("-->",count)
      count += 1
    #
    return v_out


  def oracle_bpe(self,):
    """ BPE merge ops with oracle rules for SUF, PREF, CIRCF """
    op_vocab = self.op_seqs
    count = 0
    for i in range(self.num_merges):
      pairs = self.get_affix_stats(op_vocab)
      if len(pairs) == 0:
        break
      best = max(pairs,key=pairs.get)
      print("merge: ",best)
      op_vocab = self.merge_affix_oracle(best,op_vocab)
      self.merge_history.append(best)
      # if count > 2:
      #   break
    #
    self.op_seqs = op_vocab

    print("total num merges: ",len(self.merge_history))

    #pdb.set_trace()

    print("...")


  #################################################################################  
