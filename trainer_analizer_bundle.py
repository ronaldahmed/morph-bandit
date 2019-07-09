import sys, os
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from tensorboardX import SummaryWriter
from torch.optim import Adam, Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import monotonic
from utils import to_cuda, \
                  fixed_var, \
                  MetricsWrap, \
                  apply_operations, \
                  PAD_ID, \
                  STOP_LABEL, \
                  SPACE_LABEL
from data_utils import dump_conllu
import subprocess as sp
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import pdb


class TrainerAnalizerBundle:
  def __init__(self,anlz_model,num_classes,args):
    self.args = args
    self.n_classes = num_classes
    self.model = anlz_model
    self.optimizer = Adam(anlz_model.parameters(), lr=args.learning_rate)
    # self.optimizer = Adadelta(anlz_model.parameters(), lr=args.learning_rate)
    self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    self.enable_gradient_clipping()
    self.cuda = to_cuda(args.gpu)
    self.writer = None
    self.scheduler = None

    if args.model_save_dir is not None:
        self.writer = SummaryWriter(os.path.join(args.model_save_dir, "logs"))
    if args.scheduler:
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', 0.1, 10, True)


  ### Adapted from AllenNLP
  def enable_gradient_clipping(self) -> None:
    clip = self.args.clip
    if clip is not None and clip > 0:
      # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
      # pylint: disable=invalid-unary-operand-type
      clip_function = lambda grad: grad.clamp(-clip, clip)
      for parameter in self.model.parameters():
        if parameter.requires_grad:
          parameter.register_hook(clip_function)


  def compute_loss(self,pred_seq,gold_labs,debug=0):
    total_loss = []
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
    batch, gold_output = bundle
    batch_size = gold_output.shape[0]
    hidden = self.model.refactor_hidden(batch_size)
    self.optimizer.zero_grad()
    pred_seq,hidden = self.model.forward(batch, hidden)
    loss = self.compute_loss(pred_seq, gold_output, debug)
    loss.backward()
    self.optimizer.step()

    return loss.item()


  def eval_batch(self,bundle,debug=0):
    self.model.eval()
    batch,gold_output = bundle
    batch_size = gold_output.shape[0]
    hidden = self.model.refactor_hidden(batch_size)
    
    with torch.no_grad():
      output,hidden = self.model.forward(batch, hidden)
      loss = self.compute_loss(output, gold_output, debug)
    return loss.item()


  def predict_batch(self,batch,score=False):
    self.model.eval()
    batch_size = batch[0].shape[0]
    hidden = self.model.refactor_hidden(batch_size)
    probs = []
    with torch.no_grad():
      if not score:
        pred,hidden = self.model.predict(batch,hidden)
      else:
        output,hidden = self.model.forward(batch,hidden)
        pred = self.model.argmax(output).view(batch_size,-1).data.cpu().numpy()
        probs,_ = torch.nn.functional.softmax(output).max(1)
        probs = probs.view(batch_size,-1).data.cpu().numpy()

    if score:
      return pred,probs
    return pred


  def eval_metrics_batch(self,trainer_lem,batch,data_vocabs,split='train',max_data=-1,
                          covered=False, dump_ops=False, output_name=None):
    """ eval lemmatizer using official script """
    cnt = 0
    stop_id = data_vocabs.vocab_oplabel.get_label_id(STOP_LABEL)
    forms_to_dump = []
    pred_lem_to_dump = []
    pred_feats_to_dump = []
    gold_lem_to_dump = []
    gold_feats_to_dump = []
    ops_to_dump = []
    # max_data = 3

    for bundle in batch.get_eval_batch():
      op_seqs,feats,forms,lemmas = bundle
      forms_to_dump.extend(forms)
      gold_lem_to_dump.extend(lemmas)
      gold_feats_to_dump.extend([[data_vocabs.get_feat_label(x) for x in sent] for sent in feats])
      
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
          # form_str = forms[i][j].replace(SPACE_LABEL," ")
          form_str = forms[i][j]

          # original not likely to have a cap S in the middle of the tok
          #   aaaaSbbbb
          # if S is only cap and is in the middle --> replace
          # else                                   --> leave orig
          
          # k = form_str.find(SPACE_LABEL)
          # if k!=-1:
          #   hypot = form_str[:k] + SPACE_LABEL.lower() + form_str[k+1:]
          #   if form_str.lower() == hypot and k!=0 and k!=len(form_str)-1:
          #     form_str = form_str.replace(SPACE_LABEL," ")


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
            # print("-----------------------------------")
            # print(optokens)
            # print(op_len)
            # pdb.set_trace()

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

      # 2. predict labels
      pred_labels = self.predict_batch(filtered_op_batch) # [bs x S]
      bs = pred_labels.shape[0]
      for i in range(bs):
        len_sent = len(forms[i])
        pred_feats_to_dump.append( [ data_vocabs.get_feat_label(x) for x in pred_labels[i,:len_sent] ] )
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
      filename += "."+self.args.exp_id

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
      f1 = 100*f1_score(gold_ids,pred_ids,average="micro")
      metrics = MetricsWrap(output_res[0],output_res[1],output_res[2],f1)

    return metrics


  def save_model(self,epoch):
    if self.args.model_save_dir is not None:
      if not os.path.exists(self.args.model_save_dir):
        os.makedirs(self.args.model_save_dir)
      model_save_file = os.path.join(self.args.model_save_dir, "{}_{}.pth".format(self.args.exp_id, epoch))
      print("Saving model to", model_save_file)
      torch.save(self.model.state_dict(), model_save_file)


  def update_summary(self,
                     step,
                     train_loss,
                     dev_loss=None,
                     train_metrics=None,
                     dev_metrics=None
                     ):
    if self.writer is not None:
      for name, param in self.model.named_parameters():
        self.writer.add_scalar("parameter_mean/" + name,
                          param.data.mean(),
                          step)
        self.writer.add_scalar("parameter_std/" + name, param.data.std(), step)
        if param.grad is not None:
            self.writer.add_scalar("gradient_mean/" + name,
                              param.grad.data.mean(),
                              step)
            self.writer.add_scalar("gradient_std/" + name,
                              param.grad.data.std(),
                              step)

      if train_metrics!=None:
        for var,name in zip([train_loss,
                             train_metrics.lem_acc,
                             train_metrics.lem_edist,
                             train_metrics.msd_acc,
                             train_metrics.msd_f1,
                             ],
                            ["loss/loss_train",
                             "acc/train_lem_acc",
                             "acc/train_lem_edist",
                             "acc/train_msd_acc"  ,
                             "acc/train_msd_f1"   ,
                             ]):
          
          self.writer.add_scalar(name, var, step)

      if dev_metrics!=None:
        for var,name in zip([dev_loss,
                             dev_metrics.lem_acc,
                             dev_metrics.lem_edist,
                             dev_metrics.msd_acc,
                             dev_metrics.msd_f1
                             ],
                            ["loss/loss_dev",
                             "acc/dev_lem_acc",
                             "acc/dev_lem_edist",
                             "acc/dev_msd_acc",
                             "acc/dev_msd_f1"
                             ]):
          
          self.writer.add_scalar(name, var, step)
    #





