import sys, os
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import monotonic
from utils import to_cuda, \
                  fixed_var, \
                  apply_operations, \
                  PAD_ID, \
                  STOP_LABEL, \
                  SPACE_LABEL
from data_utils import dump_conllu
import subprocess as sp

import pdb

class MetricsWrap:
  def __init__(self,acc,dist,msd_acc,msd_f1):
    self.lem_acc = acc
    self.lem_edist = dist
    self.msd_acc = msd_acc
    self.msd_f1 = msd_f1


class TrainerAnalizer:
  def __init__(self,anlz_model,num_classes,args):
    self.args = args
    self.n_classes = num_classes
    self.model = anlz_model
    self.optimizer = Adam(model.parameters(), lr=args.learning_rate)
    self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    self.enable_gradient_clipping()
    self.cuda = to_cuda(args.gpu)
    self.writer = None
    self.scheduler = None
    self.freeze_lemma()

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


  def repackage_hidden(self,h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
      return tuple(self.repackage_hidden(v) for v in h)

  def flush_hidden(self,h):
    if isinstance(h, torch.Tensor):
      return h.data.zero_()
    else:
      return tuple(self.flush_hidden(v) for v in h)

  def slice(self,h,bs):
    if isinstance(h, torch.Tensor):
      return h[:,:bs,:]
    else:
      return tuple(self.slice(v,bs) for v in h)


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

      
  def train_batch(self, batch, gold_output, debug=0):
    """Train on one batch of sentences """
    self.model.train()

    batch_size = gold_output[0].shape[0]
    #batch_size = len(gold_output[0])
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    hidden = self.repackage_hidden(hidden)
    self.optimizer.zero_grad()
    pred_seq,hidden = self.model.forward(batch, hidden)
    loss = self.compute_loss(pred_seq, gold_output, debug)
    loss.backward()
    self.optimizer.step()    

    return loss.item()


  def eval_batch(self,batch,gold_output,debug=0):
    self.model.eval()
    batch_size = gold_output[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    
    with torch.no_grad():
      hidden = self.repackage_hidden(hidden) # ([]
      output,hidden = self.model.forward(batch, hidden)
      loss = self.compute_loss(output, gold_output, debug)
    return loss.item()


  def predict_batch(self,batch,trainer_lem):
    self.model.eval()

    
    batch_size = batch[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    
    with torch.no_grad():
      hidden = self.repackage_hidden(hidden) # ([]
      pred,hidden = self.model.predict(batch_ops,hidden)
    return pred

  def get_batch_op(self,batch,data_vocabs,trainer_lem):
    batch_ops = trainer_lem.model.predict_batch(batch) # Sx[bs x W]
    cleaned = []
    # clean batch
    for w in batch_ops:
      #


  def eval_metrics_batch(self,batch,data_vocabs,split='train',max_data=-1,covered=False):
    """ eval lemmatizer using official script """
    cnt = 0
    stop_id = data_vocabs.vocab_oplabel.get_label_id(STOP_LABEL)
    forms_to_dump = []
    pred_lem_to_dump = []
    gold_lem_to_dump = []

    for op_seqs,feats,forms,lemmas in batch.get_eval_batch():
      predicted = self.predict_batch(op_seqs)
      predicted = batch.restore_batch(predicted)

      forms_to_dump.extend(forms)
      gold_lem_to_dump.extend(lemmas)
      # get op labels & apply oracle 
      for i,sent in enumerate(predicted):
        sent = predicted[i]
        # forms_to_dump.append( [data_vocabs.vocab_forms.get_label_name(x) \
        #                         for x in forms[i]] )
        # gold_lem_to_dump.append( [data_vocabs.vocab_lemmas.get_label_name(x) \
        #                         for x in lemmas[i]] )
        pred_lemmas = []
        len_sent = len(forms[i]) # forms and lemmas are not sent-padded
        for j in range(len_sent):
          w_op_seq = sent[j]
          # form_str = data_vocabs.vocab_forms.get_label_name(forms[i][j])
          form_str = forms[i][j].replace(SPACE_LABEL," ")
          if sum(w_op_seq)==0:
            pred_lemmas.append(form_str.lower())
            continue
            
          if stop_id in w_op_seq:
            _id = np.where(np.array(w_op_seq)==stop_id)[0][0]
            w_op_seq = w_op_seq[:_id+1]
          optokens = [data_vocabs.vocab_oplabel.get_label_name(x) \
                        for x in w_op_seq if x!=PAD_ID]
          pred_lem = apply_operations(form_str,optokens).replace(SPACE_LABEL," ")
          pred_lemmas.append(pred_lem)
        #
        if len(pred_lemmas)==0:
          pdb.set_trace()
        pred_lem_to_dump.append(pred_lemmas)
      #
      #pdb.set_trace()

      cnt += op_seqs[0].shape[0]
      if max_data!=-1 and cnt > max_data:
        break
    #
    filename = ""
    if   split=='train':
      filename = self.args.train_file
    elif split=='dev':
      filename = self.args.dev_file
    elif split=='test':
      filename = self.args.test_file
    
    dump_conllu(filename + ".conllu.gold",forms=forms_to_dump,lemmas=gold_lem_to_dump)
    dump_conllu(filename + ".conllu.pred",forms=forms_to_dump,lemmas=pred_lem_to_dump)

    if covered:
      return -1,-1

    else:
      pobj = sp.run(["python3","2019/evaluation/evaluate_2019_task2.py",
                     "--reference", filename + ".conllu.gold",
                     "--output"   , filename + ".conllu.pred",
                    ], capture_output=True)
      output_res = pobj.stdout.decode().strip("\n").strip(" ").split("\t")  
      output_res = [float(x) for x in output_res]
      return output_res[:2]


  def save_model(self,epoch):
    if self.args.model_save_dir is not None:
      if not os.path.exists(self.args.model_save_dir):
        os.makedirs(self.args.model_save_dir)
      model_save_file = os.path.join(self.args.model_save_dir, "{}_{}.pth".format("segm", epoch))
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

      for var,name in zip([train_loss,dev_loss,
                           train_metrics.lem_acc,dev_metrics.lem_acc,
                           train_metrics.lem_edist,dev_metrics.lem_edist,
                           train_metrics.msd_acc,dev_metrics.msd_acc,
                           train_metrics.msd_f1,dev_metrics.msd_f1
                           ],
                          ["loss/loss_train","loss/loss_dev",
                           "acc/train_lem_acc","acc/dev_lem_acc",
                           "acc/train_lem_edist","acc/dev_lem_edist",
                           "acc/train_msd_acc","acc/dev_msd_acc",
                           "acc/train_msd_f1","acc/dev_msd_f1"
                           ]):
        #if isinstance(dev_loss, torch.Tensor):
        if var != None:
          self.writer.add_scalar(name, var, step)
    #





