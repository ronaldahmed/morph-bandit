import sys, os
import torch
import numpy as np
from trainer_analizer_bundle import TrainerAnalizerBundle
from utils import to_cuda, \
                  fixed_var, \
                  MetricsWrap, \
                  apply_operations, \
                  PAD_ID, \
                  STOP_LABEL, \
                  SPACE_LABEL, \
                  SOS, EOS, \
                  UNK_TOKEN, EMPTY
from data_utils import dump_conllu
import unicodedata
import subprocess as sp
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import pdb


class TrainerAnalizerSeq(TrainerAnalizerBundle):
  def __init__(self,anlz_model,num_classes,args):
    super(TrainerAnalizerSeq, self).__init__(anlz_model,num_classes,args)


  def compute_loss(self,pred_seq,gold_labs,debug=0):
    batch_size = gold_labs.shape[0]

    mask = (gold_labs!=PAD_ID).float() # [bs,S]
    sum_mask = mask.sum(1)
    sum_mask[sum_mask==0] = 1

    gold_labs = gold_labs.view(-1)

    loss = self.loss_function(pred_seq,gold_labs)       # [bs*S]
    loss = ((loss.view(batch_size,-1)*mask).sum(1) / sum_mask).sum()  # [1]
    
    return loss
      

  def train_batch(self, bundle, debug=0):
    """Train on one batch of sentences """
    self.model.train()
    batch, input_dec, tgt_dec = bundle
    batch_size = batch[0].shape[0]
    hidden = self.model.refactor_hidden(batch_size)
    self.optimizer.zero_grad()
    pred_sent_seqs = self.model.forward(batch,input_dec,hidden)

    total_loss = 0
    for dec_out,gold in zip(pred_sent_seqs,tgt_dec):
      total_loss += self.compute_loss(dec_out,gold)

    total_loss.backward()
    self.optimizer.step()
    
    return total_loss.item()


  def eval_batch(self,bundle,debug=0):
    self.model.eval()
    batch, input_dec, tgt_dec = bundle
    batch_size = batch[0].shape[0]
    hidden = self.model.refactor_hidden(batch_size)
    
    with torch.no_grad():
      pred_sent_seqs = self.model.forward(batch,input_dec,hidden)
      total_loss = 0
      for dec_out,gold in zip(pred_sent_seqs,tgt_dec):
        total_loss += self.compute_loss(dec_out,gold).item()
    return total_loss


  """ greedy decoder
  """
  def predict_batch(self,batch,token_ids=[]):
    self.model.eval()
    batch_size = batch[0].shape[0]
    hidden = self.model.refactor_hidden(batch_size)
    sos_id = token_ids[0] # expected to be this way

    dec_outputs = []
    with torch.no_grad():
      encoder_output_hidden = self.model.encoder.forward(batch,hidden)
      
      for op_enc_seq,enc_hid in encoder_output_hidden:
        dec_input = self.cuda(torch.LongTensor([sos_id]*batch_size).view(batch_size,1))
        feat_seq = []
        for t in range(self.args.max_feat_seq):
          output,hid,attn_w = self.model.decoder.forward(dec_input,enc_hid,op_enc_seq)
          # output : [bs,nfeats]
          logits = output.div(self.args.temperature).exp()
          dec_input = torch.multinomial(logits, 1) # [bs,1]
          dec_input = dec_input.detach()
          feat_seq.append(dec_input)
        #
        feat_seq = torch.cat(feat_seq,1).cpu().numpy()
        dec_outputs.append(feat_seq)
      #
    return dec_outputs


  def eval_metrics_batch(self,trainer_lem,batch,data_vocabs,split='train',max_data=-1,
                          covered=False, dump_ops=False, output_name=None):
    """ eval lemmatizer using official script """
    cnt = 0
    stop_id = data_vocabs.vocab_oplabel.get_label_id(STOP_LABEL)
    sos_id = data_vocabs.vocab_feats.get_label_id(SOS)
    eos_id = data_vocabs.vocab_feats.get_label_id(EOS)
    unk_id = data_vocabs.vocab_feats.get_label_id(UNK_TOKEN)
    empty_id = data_vocabs.vocab_feats.get_label_id(EMPTY)
    nfeats = data_vocabs.get_feat_vocab_size()

    forms_to_dump = []
    pred_lem_to_dump = []
    pred_feats_to_dump = []
    gold_lem_to_dump = []
    gold_feats_to_dump = []
    ops_to_dump = []
    # max_data = 3

    for bundle in batch.get_eval_batch():  
      op_seqs,forms,lemmas,feats = bundle
      # feats is in [[bs x n_op] x S]

      forms_to_dump.extend(forms)
      gold_lem_to_dump.extend(lemmas)
      gold_feats_to_dump.extend([[ ";".join([data_vocabs.get_feat_label(y) for y in x[1:-1]]) \
                                              for x in sent] for sent in feats])

      filtered_op_batch = []             # bs x [ S x W ]

      # 1. predict operation sequence
      predicted_orig = trainer_lem.predict_batch(op_seqs,start=True) # Sx[ bs x W ]
      predicted = batch.restore_batch(predicted_orig)     # bs x [ SxW ]
      #    get op labels & apply oracle 
      for i,sent in enumerate(predicted):
        sent = predicted[i]
        pred_lemmas = []
        filt_op_sent = []
        op_sent = [] # to dump
        len_sent = len(forms[i]) # forms and lemmas are not sent-padded
        for j in range(len_sent):
          w_op_seq = sent[j]
          form_str = forms[i][j]
          if sum(w_op_seq)==0:
            pred_lemmas.append(form_str.lower())
            continue
          if stop_id in w_op_seq:
            _id = np.where(np.array(w_op_seq)==stop_id)[0][0]
            w_op_seq = w_op_seq[:_id+1]
          optokens = [data_vocabs.vocab_oplabel.get_label_name(x) \
                        for x in w_op_seq if x!=PAD_ID]
          pred_lem,op_len = apply_operations(form_str,optokens)

          # pred_lem = pred_lem.replace(SPACE_LABEL," ")
          pred_lemmas.append(pred_lem)
          filt_op_sent.append( w_op_seq[:op_len+1].tolist() ) # discarded the stop_id
          if dump_ops:
            op_sent.append(" ".join([x for x in optokens[1:op_len+1] if not x.startswith("STOP") and not x.startswith("START")]) )

        #
        if len(pred_lemmas)==0:
          pdb.set_trace()
        pred_lem_to_dump.append(pred_lemmas)
        filtered_op_batch.append(filt_op_sent)
        ops_to_dump.append(op_sent)
      #

      #  rebatch op seqs
      padded = batch.pad_data_per_batch(filtered_op_batch,[np.arange(len(filtered_op_batch))])
      filtered_op_batch = batch.invert_axes(padded,np.arange(len(filtered_op_batch))) # Sx[ bs x W ]

      # 2. predict labels / FINE-SEQ version
      pred_labels_seq = self.predict_batch(filtered_op_batch,[sos_id]) # [[bs x Feat] x S]
      bs = pred_labels_seq[0].shape[0]

      ## cases to filter
      # SOS, unk, empty, duplicates
      ## stop criteria
      # stop, EOS, PAD

      for i in range(bs):
        sent = []
        len_sent = len(forms[i])
        for j in range(len_sent):
          if len(forms[i][j])==1 and \
             unicodedata.category(forms[i][j]).startswith("P"):
            sent.append(EMPTY)
            continue

          feat_seq = []
          feats_found = np.zeros(nfeats,dtype=bool)
          fs_ji = pred_labels_seq[j][i,:]
          for k in range(self.args.max_feat_seq):
            if fs_ji[k] in [sos_id,unk_id,empty_id]:
              continue
            if fs_ji[k] in [stop_id,eos_id,PAD_ID]:
              break
            if not feats_found[fs_ji[k]]:
              feat_seq.append(data_vocabs.get_feat_label(fs_ji[k]))
              feats_found[fs_ji[k]] = True
          if len(feat_seq)==0:
            feat_seq = [EMPTY]
          sent.append(";".join(feat_seq))
        #
        pred_feats_to_dump.append(sent)
      #
      cnt += op_seqs[0].shape[0]
      if max_data!=-1 and cnt > max_data:
        break
    #END-FOR-BATCH

    filename = ""
    if output_name!=None:
      filename = output_name
    else:
      if   split=='train':
        filename = self.args.train_file
      elif split=='dev':
        filename = self.args.dev_file
      elif split=='test':
        filename = self.args.test_file
      filename += ".anlz.fine-seq"

    ops_to_dump = ops_to_dump if dump_ops else None
    dump_conllu(filename + ".conllu.gold",forms=forms_to_dump,lemmas=gold_lem_to_dump,feats=gold_feats_to_dump)
    dump_conllu(filename + ".conllu.pred",forms=forms_to_dump,lemmas=pred_lem_to_dump,feats=pred_feats_to_dump,ops=ops_to_dump)

    if covered:
      return MetricsWrap(-1,-1,-1,-1)

    pobj = sp.run(["python3","2019/evaluation/evaluate_2019_task2.py",
                   "--reference", filename + ".conllu.gold",
                   "--output"   , filename + ".conllu.pred",
                  ], capture_output=True)
    output_res = pobj.stdout.decode().strip("\n").strip(" ").split("\t")  

    output_res = [float(x) for x in output_res]

    metrics = None
    if self.args.eval_mode == "bundle":
      metrics = MetricsWrap(output_res[0],output_res[1],output_res[2],output_res[3])

    elif self.args.eval_mode == "fine":
      nw = sum([len(x) for x in gold_feats_to_dump])
      gold_ids = np.zeros([nw,data_vocabs.get_feat_vocab_size()])
      pred_ids = np.zeros([nw,data_vocabs.get_feat_vocab_size()])
      k = 0
      for gold_feat_sent,pred_feat_sent in zip(gold_feats_to_dump,pred_feats_to_dump):
        for gf,pf in zip(gold_feat_sent,pred_feat_sent):
          gold_ids[k,[data_vocabs.vocab_feats.get_label_id(x) \
                      for x in gf.split(";")]] = 1
          pred_ids[k,[data_vocabs.vocab_feats.get_label_id(x) \
                      for x in pf.split(";")]] = 1
          k += 1
      #
      f1 = f1_score(gold_ids,pred_ids,average="micro") # average is pessimistic
      metrics = MetricsWrap(output_res[0],output_res[1],output_res[2],f1)

    return metrics


