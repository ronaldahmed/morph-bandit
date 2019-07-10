from label_dictionary import LabelDictionary
import numpy as np
import pdb
from utils import oplabel_pat, apply_operations, \
                  to_cuda, fixed_var, \
                  PAD_ID, UNK_TOKEN, PAD_TOKEN, EOS, SOS
import torch
import copy


def dump_conllu(filename,forms,lemmas=None,feats=None,ops=None):
  nsents = len(forms)
  outfile = open(filename,'w')
  for idx_sent in range(nsents):
    for i,w in enumerate(forms[idx_sent]):
      cols = [str(i+1)] + ["_"]*9
      cols[1] = w
      try:
        if lemmas!=None: cols[2] = lemmas[idx_sent][i]
        if feats!=None:  cols[5] = feats[idx_sent][i]
        if ops!=None: cols[9] = ops[idx_sent][i]
        print("\t".join(cols),file=outfile)

      except:
        pdb.set_trace()

    print("",file=outfile)
  #
  #print("\n%s written!" % filename)



class DataWrap:
  def __init__(self,sents,feats,forms,lemmas,split):
    self.ops = sents
    self.feats = feats
    self.forms = forms
    self.lemmas = lemmas
    self.split = split

  def get_num_instances(self,):
    return len(self.forms)


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

  def get_feat_vocab_size(self,):
    return len(self.vocab_feats)

  def get_feat_label(self,ids):
    return self.vocab_feats.get_label_name(ids)

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
    ## deactivate when ml1-a1 ://
    # if self.args.tagger_mode=="fine-seq":
    sos_id = self.vocab_feats.add(SOS)
    eos_id = self.vocab_feats.add(EOS)
    self.vocab_lemmas = init_labdict(self.vocab_lemmas)
    self.vocab_forms  = init_labdict(self.vocab_forms)

    for line in open(self.args.train_file,'r'):
      line = line.strip('\n')
      if line=='': continue
      # w,lem,feats,ops = line.split("\t")
      comps = line.split("\t")
      w,lem,feats = comps[:3]
      op_seq = comps[3:]

      # op_seq = ops.split(" ")
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
      if self.args.tagger_mode=="bundle":
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
      # w,lem,feats,ops = line.split("\t")
      # op_seq = ops.split(" ")
      comps = line.split("\t")
      w,lem,feats = comps[:3]
      op_seq = comps[3:]
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
      # lem_sent.append(self.vocab_lemmas.get_label_id(lem))
      # form_sent.append(self.vocab_forms.get_label_id(w))
      lem_sent.append(lem)
      form_sent.append(w)
      if self.args.tagger_mode=="bundle":
        label.append(self.vocab_feats.get_label_id(feats))
      else:
        sos = self.vocab_feats.get_label_id(SOS)
        eos = self.vocab_feats.get_label_id(EOS)
        label.append([sos] + [self.vocab_feats.get_label_id(feat) for feat in feats.split(";")] + [eos])
    #
    return DataWrap(sents,labels,forms,lemmas,split)


class BatchBase:
  def __init__(self,data,args):
    batch_size = args.batch_size
    gpu = args.gpu
    self.in_mode = args.in_mode
    self.tag_mode = args.tagger_mode
    self.size = batch_size
    self.stop_id = data.ops[0][-1][-1] # last token: STOP.
    self.sents = self.strip_stop(data.ops)
    self.lemmas = data.lemmas
    self.forms = data.forms
    self.feats = data.feats
    self.cuda = to_cuda(gpu)
    N = len(self.sents)
    idx = list(range(N))
    if data.split == "train":
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

  def pad_data_per_batch(self,sents,list_batch_ids=None):
    new_sents = copy.deepcopy(sents)
    if list_batch_ids is None:
      list_batch_ids = self.sorted_ids_per_batch
    for batch_ids in list_batch_ids:
      max_sent_len = max([ len(sents[x]) for x in batch_ids]) # to pad sents
      max_wop_lens = []
      for i in range(max_sent_len):
        max_wop_len = max( [ len(new_sents[x][i]) for x in batch_ids if len(new_sents[x])>i ] ) # at least one holds
        max_wop_lens.append(max_wop_len)
      #
      
      for idx in batch_ids:
        # pad the op sequence, sent still same len
        new_sent = [self.right_pad(w,max_wop_lens[i],PAD_ID) for i,w in enumerate(new_sents[idx])]
        extra_words = [[PAD_ID]*x for x in max_wop_lens[len(new_sent):]]
        new_sents[idx] = new_sent + extra_words
      #
    #
    return new_sents

  """
  Handles Feature bundle tagging, i.e. one meta-tag per token
  labs[i] = 'V;3p;Pl'
  """
  def pad_labels_bundle(self,labs,list_batch_ids=None):
    if list_batch_ids is None:
      list_batch_ids = self.sorted_ids_per_batch
    for batch_ids in list_batch_ids:
      max_sent_len = max([ len(labs[x]) for x in batch_ids]) # to pad sents
      for idx in batch_ids:
        labs[idx] = self.right_pad(labs[idx],max_sent_len,PAD_ID)
    return labs

  """
  Handles list of indiv features per token, e.g. for seq decoding of features
  labs[i] = ['V','3p','Pl']
  """
  def pad_labels_per_batch(self,labs,list_batch_ids=None):
    if   self.tag_mode == "bundle":
      return self.pad_labels_bundle(labs,list_batch_ids)
    elif self.tag_mode == "fine-seq":
      return self.pad_data_per_batch(labs,list_batch_ids)


  def invert_axes(self,sequence,idxs,_eval=False):
    """ [bs,S,W] -> Sx[tensor(bs,W)] 
        padded sequence assumed
    """
    new_seq = []
    S = len(sequence[idxs[0]])
    W = len(sequence[idxs[0]][0])
    for i in range(S):
      if _eval:
        w_i = [[sequence[idx][i][0]] for idx in idxs]
      else:
        w_i = [sequence[idx][i] for idx in idxs]
      new_seq.append( self.cuda(fixed_var(torch.LongTensor(w_i))) )
      # new_seq.append(w_i)

    return new_seq


  def restore_batch(self,batch):
    """ Sx[np(bs,W)] -> [bs,S,W] """
    new_seq = []
    S = len(batch) # len of sentence
    bs = batch[0].shape[0] # batch size
    for i in range(bs):
      s_i = [ batch[j][i,:] for j in range(S)]
      new_seq.append(s_i)
    return new_seq


  def batchify(self,seq):
    batches = []
    for batch_ids in self.sorted_ids_per_batch:
      ops = self.invert_axes(seq,batch_ids)
      batches.append(ops)
    return batches



class BatchSegm(BatchBase):
  def __init__(self, data, args):
    super(BatchSegm, self).__init__(data,args)
    self.tgt_sents = [] # LM-like training
    self.build_gold_lm_seq()
    self.sents = self.pad_data_per_batch(self.sents)
    self.tgt_sents = self.pad_data_per_batch(self.tgt_sents)
    # self.src_batches = self.batchify(self.sents)
    # self.tgt_batches = self.batchify(self.tgt_sents)
    

  def build_gold_lm_seq(self,):
    """ call before padding """
    self.tgt_sents  = []
    for sent in self.sents:
      self.tgt_sents.append( [w[1:]+[self.stop_id] for w in sent ] )
    #

  def get_batch(self,shuffle=True):
    # batch_ids = np.arange(len(self.sorted_ids_per_batch))
    # if shuffle:
    #   np.random.shuffle(batch_ids)
    # for _id in batch_ids:
    #   yield self.src_batches[_id],self.tgt_[_id]
    if shuffle:
      np.random.shuffle(self.sorted_ids_per_batch)
    for batch_ids in self.sorted_ids_per_batch:
      ops = self.invert_axes(self.sents,batch_ids)
      tgt_ops = self.invert_axes(self.tgt_sents,batch_ids)
      yield ops,tgt_ops

  def get_eval_batch(self):
    for _id,batch_ids in enumerate(self.sorted_ids_per_batch):
      ops = self.invert_axes(self.sents,batch_ids,_eval=True)
      forms = [self.forms[idx] for idx in batch_ids]
      lemmas = [self.lemmas[idx] for idx in batch_ids]
      yield ops,forms,lemmas



# Wrapper to direct correct implementation
def BatchAnalizer(data, args):
  if   args.tagger_mode == "bundle":
    return BatchAnalizerBundle(data,args)
  elif args.tagger_mode == "fine-seq":
    return BatchAnalizerSeq(data,args)


### Architecture specific batchers

class BatchAnalizerBundle(BatchBase):
  def __init__(self, data, args):
    super(BatchAnalizerBundle, self).__init__(data,args)
    self.sents = self.pad_data_per_batch(self.sents)
    self.labels = self.pad_labels_per_batch(data.feats)


  def get_batch(self,shuffle=True):
    if shuffle:
      np.random.shuffle(self.sorted_ids_per_batch)
    for batch_ids in self.sorted_ids_per_batch:
      ops = self.invert_axes(self.sents,batch_ids)
      labels = self.cuda(fixed_var(
                      torch.LongTensor( np.array([self.labels[idx] for idx in batch_ids]) )))
      yield ops,labels


  def get_eval_batch(self):
    for _id,batch_ids in enumerate(self.sorted_ids_per_batch):
      ops = self.invert_axes(self.sents,batch_ids,_eval=True)
      forms = [self.forms[idx] for idx in batch_ids]
      lemmas = [self.lemmas[idx] for idx in batch_ids]
      labels = self.cuda(fixed_var(
                    torch.LongTensor( np.array([self.labels[idx] for idx in batch_ids]) )))
      yield ops,labels,forms,lemmas



class BatchAnalizerSeq(BatchBase):
  def __init__(self, data, args):
    super(BatchAnalizerSeq, self).__init__(data,args)
    self.sents = self.pad_data_per_batch(self.sents)
    src,tgt = self.build_gold_decoded_labels(data.feats)
    self.input_labels = self.pad_labels_per_batch(src)
    self.tgt_labels = self.pad_labels_per_batch(tgt)


  def build_gold_decoded_labels(self,feats):
    input_decoder = []
    tgt_decoder = []
    for sent in feats:
      new_in_sent = []
      new_tgt_sent = []
      for bundle in sent:
        new_in_sent.append(bundle[:-1])
        new_tgt_sent.append(bundle[1:])
      input_decoder.append(new_in_sent)
      tgt_decoder.append(new_tgt_sent)
    return input_decoder,tgt_decoder


  def get_batch(self,shuffle=True):
    if shuffle:
      np.random.shuffle(self.sorted_ids_per_batch)
    for batch_ids in self.sorted_ids_per_batch:
      ops = self.invert_axes(self.sents,batch_ids)
      src = self.invert_axes(self.input_labels,batch_ids)
      tgt = self.invert_axes(self.tgt_labels,batch_ids)
      yield ops,src,tgt


  def get_eval_batch(self):
    for _id,batch_ids in enumerate(self.sorted_ids_per_batch):
      ops = self.invert_axes(self.sents,batch_ids,_eval=True)
      forms = [self.forms[idx] for idx in batch_ids]
      lemmas = [self.lemmas[idx] for idx in batch_ids]
      feats = [self.feats[idx] for idx in batch_ids]
      src = self.invert_axes(self.input_labels,batch_ids)
      tgt = self.invert_axes(self.tgt_labels,batch_ids)
      
      yield ops,forms,lemmas,feats
