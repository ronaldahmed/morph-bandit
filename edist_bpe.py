import os, sys
import glob as gb
import numpy as np
import pdb
import pickle

from collections import defaultdict
from label_dictionary import LabelDictionary
from utils import *

from op_typeins import *



#########################################################################################


class EDistBPE:
  def __init__(self,_num_merges=50,_inflector=False):
    # direction of analysis
    #  lem->form : inflector True
    #  form->lem : inflector False
    self.inflector = _inflector

    # operation dictionary
    self.op_dict = LabelDictionary()
    self.op_lexicon = {}
    self.populate_op_ids()

    # keep track of data for counts
    self.op_seqs = []
    self.label_lid = []
  
    # number of BPE merges / ops
    self.num_merges = _num_merges
    self.merge_history = []
    self.merge_bigrams = {}
    # internal flafs
    self._encoding = False


  def populate_op_ids(self):
    # order here coincides with EditDist scoring sequence
    for _op in [DEL,INS,SUBS,TRSP,SKIP]:
      self.op_dict.add(_op)


  def train(self,vocab):
    """
      train BPE encoder based on edit-distance operations 
      vocab: [ (form,lemma,feat)_i ]
      merge_file: filename where to output licensed merges
    """
    op_seqs = []
    for lem,forms in vocab.items():
      for form in forms:
        ops = self.get_primitive_actions(lem,form)
        op_seqs.append(ops)
    count = 0

    for i in range(self.num_merges):
      pairs = self.get_affix_stats(op_seqs)
      if len(pairs) == 0:
        break
      best = max(pairs,key=pairs.get)
      print("merge: ",best)
      op_seqs = self.merge_affix_oracle(best,op_seqs)
      self.merge_history.append([(x.name,x.segment) for x in best])
    #
    print("total num merges: ",len(self.merge_history))
    self.merge_bigrams = self.build_merge_bigrams(self.merge_history)


  def dump_merge_file(self,tbname):
    if not os.path.exists("data/"+tbname):
      os.makedirs("data/"+tbname)
    outfile = open("data/%s/merges" % tbname,'w')
    for op1,op2 in self.merge_history:
      print("%s-%s\t%s-%s" % (op1[0],op1[1],op2[0],op2[1]),file=outfile)


  def load_merge_file(self,filename):
    merges = []
    for line in open(merge_file,'r'):
      line = line.strip("\n")
      if line=="": continue
      op1,op2 = line.strip("\t")
      merges.append([op1.split("-"),op2.split("-")])
    return self.build_merge_bigrams(merges)


  def build_merge_bigrams(self,merge_hist):
    """ builds merge lookup data struct 
        {name: seg1+seg2} -> p[ins][s1] -> s2
    """
    merge_dict = defaultdict(set)
    for (op1,op2) in merge_hist:
      n1,s1 = op1
      n2,s2 = op2 # n1=n2, by def
      merge_dict[n1].add(s1+s2)
    #
    return merge_dict


  def encode(self,data,filename,merge_file=None):
    """ Encodes sequence of (form,lema) tupples into operation sequences
    """
    if merge_file!=None:
      self.merge_bigrams = self.load_merge_file(merge_file)
    
    outfile = open(filename,'w')
    stop_tok = STOP+"."+STOP+"-</>"
    count = 0
    for sent in data:
      for form,lemma,feat in sent:
        ops = self.get_primitive_actions(lemma.lower(),form.lower())
        merged_ops = self.merge_ops_sent(ops)
        op_tokens = self.oracle_sequencer(merged_ops)
        orig = lemma.lower() if self.inflector else form.lower()
        start_tok = "%s.%s-%s" % (START,START,orig)
        op_tokens = [start_tok] + op_tokens + [stop_tok]
        op_tok_str = " ".join(op_tokens)
        print("%s\t%s\t%s\t%s" % (form,lemma,feat,op_tok_str),file=outfile)

        if count % 10000 == 0:
          print("->",count)
        count += 1
      #
      print("",file=outfile)
    #


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


  def oracle_sequencer(self,ops):
    if ops[0].name==STOP or ops==[]:
      return ops
    
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
              i>0 and ops[i-1].name==DEL and ops[i].name==DEL and ops[i-1].pos + len(ops[i-1].segment) == ops[i].pos,
              i>0 and ops[i-1].name==DEL and ops[i].name!=INS and ops[i-1].pos + len(ops[i-1].segment) == ops[i].pos,
              # i>0 and ops[i-1].pos==ops[i].pos,
              # i>0 and ops[i-1].name!=INS and ops[i-1].pos!=ops[i].pos and ops[i-1].pos + len(ops[i-1].segment) == ops[i].pos,
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
      if any([ ops[i].name==INS and ops[i].pos > len_form,
               # ops[i].name==DEL and ops[i].pos == len_form,
               ops[i].name!=INS and ops[i].pos-1 + len(ops[i].segment)==len_form,
               i<n_ops-1 and ops[i].name!=INS and ops[i+1].name==DEL and ops[i].pos + len(ops[i].segment) == ops[i+1].pos,
               i<n_ops-1 and ops[i].name==DEL and ops[i+1].name==DEL and ops[i].pos + len(ops[i].segment) == ops[i+1].pos,
               # i<n_ops-1 and ops[i].pos==ops[i+1].pos,
               # i<n_ops-1 and ops[i].name!=INS and ops[i].pos!=ops[i+1].pos and ops[i].pos + len(ops[i].segment) == ops[i+1].pos,
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

    if prefs_cnt >= sufs_cnt:
      print("prefs > sufs!!")
      pdb.set_trace()

    # d) split ops in prefix - mid - suffix ops
    prefs = [x for x in ops[:prefs_cnt+1] if x.name!=SKIP]
    suffs = [x for x in ops[sufs_cnt:] if x.name!=SKIP]
    mids  = [x for x in ops[prefs_cnt+1:sufs_cnt] if x.name!=SKIP]

    # e) reverse ins/del ops accordingly

    def reverse_by_blocks(seq,mode='suff'):
      """ mode: suff, pref
      """
      if seq==[]: return []

      buff = []
      ins = []
      cnt = 0
      # seq.append(Operation(STOP,-1,seq[0].form,True))
      # n_seq = len(seq)
      for i,op in enumerate(seq):
        if op.name==INS:
          ins.append(op)
        else:
          buff.append(op)
          # if op.name == DEL:
          #   cnt += 1
          # else:
          #   if cnt>0 and mode=="suff":
          #     buff = list(reversed(buff))
          #   cnt = 0
          # if i < n_seq-1:
      #
      if mode=="suff":
        buff = list(reversed(buff)) + ins
      else:
        buff += list(reversed(ins))
      return buff

    # e.1. prefix: reverse INS  
    prefs = reverse_by_blocks(prefs,'pref') # <---
    suffs = reverse_by_blocks(suffs,'suff') # --->
  
    # f) relitivize positions 
    discounter = 0
    for x in prefs:
      if x.name==INS:
        discounter += len(x.segment)
      if x.name==DEL:
        discounter -= len(x.segment)
    for i in range(len(mids)):
      mids[i].pos += discounter
      if mids[i].name==INS:
        discounter += len(mids[i].segment)
      if mids[i].name==DEL:
        discounter -= len(mids[i].segment)
      if mids[i].pos < 1:
        print("Bad discounting!!")
        pdb.set_trace()

    # g) encode operation & aggregate
    sent = ["%s.%s-%s" % (op.name,PREF_POS ,op.segment) for op in prefs] + \
           ["%s._%d_-%s" % (op.name,op.pos,op.segment) for op in mids] + \
           ["%s.%s-%s" % (op.name,SUFF_POS ,op.segment) for op in suffs]
    

    # if len(prefs)>1:
    #   print(prefs)
    #   pdb.set_trace()


    return sent


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


  def merge_ops_sent(self,seq):
    def is_contiguous(op1,op2):
      if op1.name==INS:
        return op1.pos == op2.pos
      return op1.pos + len(op1.segment) == op2.pos

    curr_seg = ""
    cnt = 1
    curr_type = "-"
    merged_ops = [seq[0]]
    for op in seq[1:]:
      if op.name==SKIP:
        merged_ops.append(op)
        continue
      if all([merged_ops[-1].name==op.name,
              merged_ops[-1].segment + op.segment in self.merge_bigrams[op.name],
              is_contiguous(merged_ops[-1],op)]):
        merged_ops[-1] = self.join_operation(merged_ops[-1],op)
      else:
        merged_ops.append(op)
    #
    # if len(merged_ops)!=len(seq):
    #   pdb.set_trace()
    return merged_ops


  # def merge_ops_sent(self,seq):
  #   """ dynamic prog algo to cover seq with min num of licensed chunks """

  #   N = len(seq)
  #   dp = MAX_INT*np.ones([N,N],dtype=int)
  #   bp = {}
  #   for i in range(N):
  #     dp[i,i] = 1
  #   curr_seg = ""
  #   cnt = 1
  #   curr_type = "-"
  #   for i,op in enumerate(seq):
  #     if curr_type!=op.name:
  #       if curr_type!=SKIP and curr_seg in self.merge_bigrams[curr_type]:
  #         dp[i-cnt+1,i] = 1
  #       curr_type = op.name
  #       curr_seg = op.segment
  #       cnt = 1
  #     else:
  #       curr_seg += op.segment
  #       cnt += 1
  #   #
  #   if cnt>1 and curr_seg in self.merge_bigrams[curr_type]:
  #     dp[N-cnt,N-1] = 1

  #   for k in range(2,N+1):
  #     for i in range(N-k+1):
  #       for j in range(i+1,i+k):
  #         # if same-name block ends, break
  #         if seq[i].name != seq[j].name:
  #           break

  #         if dp[i,j-1] + dp[j,i+k-1] < dp[i,i+k-1]:
  #           dp[i,i+k-1] = dp[i,j-1] + dp[j,i+k-1]
  #           bp[i,i+k-1] = j
  #         #
  #   #
  #   def backtrack(i,j):
  #     if j==i+1:
  #       return seq[i:j+1]
  #     elif i==j:
  #       return [seq[i]]
  #     if (i,j) not in bp:
  #       if dp[i,j]==1:
  #         seg = "".join([op.segment for op in seq[i:j+1]])
  #         return [Operation(seq[i].name,seq[i].pos,seg,seq[i].form)]
  #       else:
  #         return seq[i:j+1]
  #     return backtrack(i,bp[i,j]) + backtrack(bp[i,j]+1,j)

  #   # pdb.set_trace()

  #   merged_ops = backtrack(0,N-1) if len(bp)!=0 else seq
  #   if len(merged_ops)!=len(seq):
  #     print("changed!!")
  #     pdb.set_trace()

    
  #   return merged_ops



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


  #################################################################################  
  
