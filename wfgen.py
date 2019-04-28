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


  def read_task2(self,tb_names):
    count = 0
    for tbname in tb_names:
      uddir = self.tbname2uddir[tbname]
      filename = "2019/task2/%s/%s-um-train.conllu" % (uddir,tbname)

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
        
        if count % 10000 == 0:
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
    op_lexicon = {}
    stop_tok = STOP+"."+STOP+"-</>"
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
        
        print(line)
        print(form,"-->",lem)
        op_tokens = self.encode_op_seq(ops)
        orig = lem if self.inflector else form
        start_tok = "%s.%s-%s" % (START,START,orig)
        op_tokens = [start_tok] + op_tokens + [stop_tok]
        
        sent.append(" ".join(op_tokens))

        _key = form + "\t" + lem if not self.inflector else lem + "\t" + form
        if _key not in op_lexicon:
          op_lexicon[_key] = op_tokens
        elif len(op_tokens) != len(op_lexicon[_key]):
          print("Diff derivations!!")
          pdb.set_trace()

        # print(op_tokens)
        # if count % 5 == 0:
        #   pdb.set_trace()
        #   print("---")
        count += 1
      #
    #
    op_lex_file = open(outfn + ".op_lex",'w')

    for _key,tokens in op_lexicon.items():
      print("%s\t%s" % (_key," ".join(tokens)),file=op_lex_file)


  def get_primitive_actions(self,str1,str2):
    """ algorithm based on DP edit-distance (Damerau–Levenshtein distance)  """
    # 1: lemma
    # 2: form
    # print(lemma,"--",form)
    if not self.inflector:
      str2,str1 = str1,str2

    n = len(str1)
    m = len(str2)
    dp = MAX_INT*np.ones([n+1,m+1],dtype=int)
    bp = {}
    for i in range(n): dp[i,0] = i
    for i in range(m): dp[0,i] = i
    bp[(0,0)] = [-1,-1,'']
    for i in range(1,n+1): bp[(i,0)] = [i-1,0,DEL]
    for i in range(1,m+1): bp[(0,i)] = [0,i-1,INS]
     
    for j in range(1,m+1):
      for i in range(1,n+1):
        if str1[i-1] == str2[j-1]:
          dp[i,j] = dp[i-1,j-1]
          bp[i,j] = [i-1,j-1,SKIP]
          # print(i,j,SKIP)
        else:
          # transposition op conds
          #                     del,       ins ,       subst        trans
          scores = np.array([dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+1,MAX_INT])

          if i>1 and j>1 and \
             str1[i-1]==str2[j-2] and str1[i-2]==str2[j-1]:
            scores[-1] = dp[i-2,j-2] + 1

          dp[i,j] = scores.min()
          action = self.op_dict.get_label_name(scores.argmin())
          # print(i,j,action,scores)

          if action==DEL:
            bp[i,j] = [i-1,j]
          elif action==INS:
            bp[i,j] = [i,j-1]
          elif action==SUBS:
            bp[i,j] = [i-1,j-1]
          elif action==TRSP:
            bp[i,j] = [i-2,j-2]
          bp[i,j].append(action)
        #
      #
    #
    # print("best score:",dp[n,m])
    u = bp[(n,m)]
    actions = []
    while u != [-1,-1,'']:
      i,j,action = u # indexes already discounted
      op = ''
      if action==INS:
        op = Operation(action,i+1,str2[j:j+1],str1,True)
      elif action==SUBS:
        op = Operation(action,i+1,str2[j:j+1],str1,True)
      elif action==DEL:
        op = Operation(action,i+1,str1[i:i+1],str1,True)
      elif action==TRSP:
        op = Operation(action,i+1,str1[i:i+2],str1,True)
      elif action==SKIP:
        op = Operation(action,i+1,str1[i:i+1],str1,True)
        # u = bp[i,j]
        # continue
      actions.append(op)
      u = bp[i,j]

    actions = sorted(reversed(actions),key=lambda x: (x.pos,x.name))

    # print(str1,"-->",str2)
    # for a in actions:
    #   print(a)
    # pdb.set_trace()
    # actions.append(Operation(STOP,n,str1,True))

    return actions


  def encode_op_seq(self,ops):
    # if ops[0].name==STOP:
    #   return ops
    if ops==[]:
      return []
    # ops = ops[:-1] # all but STOP
    n_ops = len(ops)
    len_form = len(ops[0].form)
  
    prefs_cnt = 0
    sufs_cnt = n_ops-1
    last_idx = 0

    # a) obtain prefix delim index
    for i in range(n_ops):
      if ops[i].name==SKIP:
        break
      if any([ops[i].pos==1,
              i>0 and ops[i-1].pos==ops[i].pos,
              i>0 and ops[i-1].name!=INS and ops[i-1].pos!=ops[i].pos and ops[i-1].pos + len(ops[i-1].segment) == ops[i].pos,
              ]):
        prefs_cnt = i
      else:
        break
    # while prefs_cnt<n_ops and ops[prefs_cnt].pos == 1:
    #   prefs_cnt += 1

    # b) obtain suffix delim index
    for i in range(n_ops-1,-1,-1):
      if ops[i].name==SKIP:
        break
      if any([ ops[i].pos >= len_form,
               i<n_ops-1 and ops[i].pos==ops[i+1].pos,
               i<n_ops-1 and ops[i].name!=INS and ops[i].pos!=ops[i+1].pos and ops[i].pos + len(ops[i].segment) == ops[i+1].pos,
               ]):
        sufs_cnt = i
      else:
        break
    # while sufs_cnt>=0 and ops[sufs_cnt].pos >= len_form and ops[sufs_cnt].pos!=1: # len = subs,del; len+1 = ins
    #   sufs_cnt -= 1
    
    # c) special cases for pref - suff interaction
    # case: both pref and suff consumed the whole sequence
    #   default: suffixation
    if sufs_cnt==0 and prefs_cnt==n_ops-1:
      prefs_cnt = -1

    # case: 1 op seq
    # default  -> suffixed
    if sufs_cnt==prefs_cnt:
      if prefs_cnt==0 and len(ops)==1:
        prefs_cnt = -1
        sufs_cnt = 0
      else:
        prefs_cnt -= 1

    # print(ops)
    # print(prefs_cnt,sufs_cnt)
    # pdb.set_trace()

    if prefs_cnt >= sufs_cnt:
      print("prefs > sufs!!")
      pdb.set_trace()

    # d) split ops in prefix - mid - suffix ops
    prefs = [x for x in ops[:prefs_cnt+1] if x.name!=SKIP]
    suffs = [x for x in ops[sufs_cnt:] if x.name!=SKIP]
    mids  = [x for x in ops[prefs_cnt+1:sufs_cnt] if x.name!=SKIP]

    # e) reverse ins/del ops accordingly
    def reverse_chunk_op(seq,name):
      if seq==[]: return []
      cnt = 0
      buff = []
      ENTRA = False
      seq.append(Operation(STOP,-1,seq[0].form,True))
      n_seq = len(seq)
      for i,op in enumerate(seq):
        if op.name==name:
          cnt += 1
        else:
          if cnt>1:
            buff = buff[:-cnt] + list(reversed(buff[-cnt:]))
            ENTRA = True
          cnt = 0
        if i < n_seq-1:
          buff.append(op)

      if ENTRA: pdb.set_trace()

      if len(buff) != n_seq-1:
        print("buff diff len!!")
        print(buff)
        pdb.set_trace()

      return buff

    # e.1. prefix: reverse INS
    try:
      prefs = reverse_chunk_op(list(reversed(prefs)),DEL) # <---
      suffs = reverse_chunk_op(suffs,DEL)                 # --->
    except:
      pdb.set_trace()

    # sent = []
    # to_inv = []
    # prefs = []
    # sufs = []
    # mid = []
    # for i,op in enumerate(ops):
    #   if op.name==SKIP:
    #     continue
    #   op_mask_code = "_A_"
    #   if i <= prefs_cnt:
    #     op_mask_code = "_A"
    #   if i >= sufs_cnt:
    #     op_mask_code = "A_"
    #   token = "%s.%s-%s" % (op.name,op_mask_code,op.segment)
    #   if op_mask_code=="A_":
    #     sufs.append(token)
    #   elif op_mask_code=="_A":
    #     prefs.append(token)
    #   else:
    #     mid.append(token)
    # ##

    # f) encode operation & aggregate
    sent = ["%s.%s-%s" % (op.name,"_A" ,op.segment) for op in prefs] + \
           ["%s.%s-%s" % (op.name,"_A_",op.segment) for op in mids] + \
           ["%s.%s-%s" % (op.name,"A_" ,op.segment) for op in suffs]
    

    if len(prefs)>1:
      print(prefs)
      pdb.set_trace()


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
    if op1.name != op2.name:
      print("Ops of diff type! not allowed")
      sys.exit(1)

    new_seg = op1.segment + op2.segment if op1.pos <= op2.pos else op2.segment + op1.segment
    new_op = Operation(op1.name,min(op1.pos,op2.pos),new_seg,op1.form,False)
    # case: two ins_pos0 ops or two ins_pos_n+1
    # assume: op1, op2 come from sorted array of ops -> op1 comes first

    return new_op


  def get_affix_stats(self,op_vocab):
    """ Counts continguos ops of the same type """
    pairs = defaultdict(int)
    for seq in op_vocab:
      if len(seq) < 3: continue # don't count stop op
      for i in range(len(seq)-2):
        # only count if adjacent and same type of OP
        op1 = seq[i]
        op2 = seq[i+1]
        # enforcing same type of op here
        if any([op1.name==SKIP,
                op1.name != op2.name,
                op1.pos != op2.pos and op1.pos + len(op1.segment) != op2.pos ]):
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
      if count % 10000 == 0:
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
