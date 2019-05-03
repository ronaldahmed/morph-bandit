from label_dictionary import LabelDictionary
import numpy as np
import pdb
from utils import oplabel_pat, apply_operations

# Special IDs
UNK_TOKEN = "*UNK*"
PAD_TOKEN = "*PAD*"
PAD_ID = 0


class DataWrap:
  def __init__(self,sents,feats,forms,lemmas):
    self.ops = sents
    self.feats = feats
    self.forms = forms,
    self.lemmas = lemmas


class DataLoaderAnalizer:
  def __init__(self,args):
    self.args = args
    self.vocab_oplabel = LabelDictionary()
    self.vocab_op_name = LabelDictionary()
    self.vocab_op_pos = LabelDictionary()
    self.vocab_op_seg = LabelDictionary()
    self.vocab_feats = LabelDictionary()
    self.vocab_lemmas = LabelDictionary()
    self.vocab_forms = LabelDictionary()
    self.load_vocab()


  def get_vocab_size(self,):
    return len(self.vocab_oplabel)


  def load_vocab(self):
    def init_labdict(_dict):
      _dict.add(PAD_TOKEN)
      _dict.add(UNK_TOKEN)
      return _dict

    if self.args.in_mode=="coarse":
      self.vocab_oplabel = init_labdict(self.vocab_oplabel)
      PAD_ID = self.vocab_oplabel.get_label_id(PAD_TOKEN)
    else:
      self.vocab_op_name = init_labdict(self.vocab_op_name)
      self.vocab_op_pos = init_labdict(self.vocab_op_pos)
      self.vocab_op_seg = init_labdict(self.vocab_op_seg)
      PAD_ID = self.vocab_op_name.get_label_id(PAD_TOKEN)
    self.vocab_feats  = init_labdict(self.vocab_feats)
    self.vocab_lemmas = init_labdict(self.vocab_lemmas)
    self.vocab_forms  = init_labdict(self.vocab_forms)

    for line in open(self.args.train_file,'r'):
      line = line.strip('\n')
      if line=='': continue
      w,lem,feats,ops = line.split("\t")
      op_seq = ops.split(" ")
      for op in op_seq:
        if self.args.in_mode=="coarse":
          self.vocab_oplabel.add(op)
        else:
          match = oplabel_pat.match(op)
          if match==None:
            print("Wrong format on operation label!!")
            pdb.set_trace()
          self.vocab_op_name.add(match.group("name"))
          self.vocab_op_pos.add(match.group("pos"))
          self.vocab_op_seg.add(match.group("seg"))
      #
      self.vocab_lemmas.add(lem)
      self.vocab_forms.add(w)
      if self.args.out_mode=="coarse":
        self.vocab_feats.add(feats)
      else:
        for feat in feats.split(";"):
          self.vocab_feats.add(feat)
    ##

  
  def load_data(self,split):
    filename = ""
    if   split=="train":
      filename = self.args.train_file
    elif split=="dev":
      filename = self.args.dev_file
    else:
      filename = self.args.test_file
    sents,labels,lemmas,forms = [],[],[],[]
    sent,label,lem_sent,form_sent = [],[],[],[]
    for line in open(filename,'r'):
      line = line.strip('\n')
      if line=='':
        sents.append(sent)
        labels.append(label)
        forms.append(form_sent)
        lemmas.append(lem_sent)
        sent = []
        label = []
        lem_sent = []
        form_sent = []
        continue
      w,lem,feats,ops = line.split("\t")
      op_seq = ops.split(" ")
      op_ids = []
      for op in op_seq:
        if self.args.in_mode=="coarse":
          op_ids.append(self.vocab_oplabel.get_label_id(op))
          
        else:
          match = oplabel_pat.match(op)
          if match==None:
            print("Wrong format on operation label!!")
            pdb.set_trace()
          op_ids.append([
            self.vocab_op_name.get_label_id(match.group("name")),
            self.vocab_op_pos.get_label_id(match.group("pos")),
            self.vocab_op_seg.get_label_id(match.group("seg")),
            ])
      #
      sent.append(op_ids)
      lem_sent.append(self.vocab_lemmas.get_label_id(lem))
      form_sent.append(self.vocab_forms.get_label_id(w))
      if self.args.out_mode=="coarse":
        label.append(self.vocab_feats.get_label_id(feats))
      else:
        label.append([self.vocab_feats.get_label_id(feat) for feat in feats.split(";")])
    #
    return DataWrap(sents,labels,forms,lemmas)



class BatchBase:
  def __init__(self,data,batch_size):
    self.size = batch_size
    self.stop_id = data.ops[0][-1][-1] # last token: STOP.
    self.sents = self.strip_stop(data.ops)
    self.lemmas = data.lemmas
    self.forms = data.forms
    N = len(self.sents)
    idx = list(range(N))
    idx.sort(key=lambda x: len(self.sents[x]),reverse=True)
    n_batches = (N//self.size) + int(N%self.size != 0)
    idx = self.right_pad(idx,n_batches*self.size,-1)

    self.sorted_ids_per_batch = np.split(np.array(idx),n_batches)
    self.sorted_ids_per_batch[-1] = np.array([x for x in self.sorted_ids_per_batch[-1] if x!=-1])


  def right_pad(self,xs, min_len, pad_element):
    """
    Appends `pad_element`s to `xs` so that it has length `min_len`.
    No-op if `len(xs) >= min_len`.
    """
    return xs + [pad_element] * (min_len - len(xs))

  def strip_stop(self,sents):
    new_sents = []
    for sent in sents:
      new_sents.append( [w[:-1] for w in sent] )
    return new_sents

  def pad_data_per_batch(self,sents):
    for batch_ids in self.sorted_ids_per_batch:
      max_sent_len = len(sents[batch_ids[0]]) # to pad sents
      max_wop_len = 0
      for idx in batch_ids:
        max_wop_len = max(max_wop_len,max([len(w)-1 for w in sents[idx]]))

      for idx in batch_ids:
        # pad the op sequence, sent still same len
        new_sent = [self.right_pad(x,max_wop_len,PAD_ID) for x in sents[idx]]
        sents[idx] = new_sent

      for idx in batch_ids:
        sents[idx] = self.right_pad(sents[idx],max_sent_len,[PAD_ID]*max_wop_len)
      #
    #
    return sents


class BatchSegm(BatchBase):
  def __init__(self, data, batch_size):
    super(BatchSegm, self).__init__(data,batch_size)
    self.tgt_sents = [] # LM-like training
    self.build_gold_lm_seq()
    self.sents = self.pad_data_per_batch(self.sents)
    self.tgt_sents = self.pad_data_per_batch(self.tgt_sents)
    

  def build_gold_lm_seq(self,):
    """ call before padding """
    self.tgt_sents  = []
    for sent in self.sents:
      self.tgt_sents.append( [w[1:]+[self.stop_id] for w in sent ] )
    #

  def get_batch(self):
    np.random.shuffle(self.sorted_ids_per_batch)
    for batch_ids in self.sorted_ids_per_batch:
      sents = [self.sents[idx] for idx in batch_ids]
      tgt_sents = [self.tgt_sents[idx] for idx in batch_ids]
      lemmas = [self.lemmas[idx] for idx in batch_ids]
      yield sents,tgt_sents

  def get_eval_batch(self):
    np.random.shuffle(self.sorted_ids_per_batch)
    for batch_ids in self.sorted_ids_per_batch:
      ops = [self.sents[idx] for idx in batch_ids]
      forms = [self.forms[idx] for idx in batch_ids]
      lemmas = [self.lemmas[idx] for idx in batch_ids]
      yield ops,forms,lemmas



class BatchAnalizer(BatchSegm):
  def __init__(self, data, batch_size):
    super(BatchBase, self).__init__(data,batch_size)
    self.sents = self.pad_data_per_batch(self.sents)
    self.labels = data.feats


  def get_batch(self):
    np.random.shuffle(self.sorted_ids_per_batch)
    for batch_ids in self.sorted_ids_per_batch:
      sents = [self.sents[idx] for idx in batch_ids]
      labels = [self.labels[idx] for idx in batch_ids]
      yield sents,labels
