import sys, os
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import monotonic
from utils import to_cuda, fixed_var, apply_operations, PAD_ID, STOP_LABEL
from data_utils import dump_conllu
import subprocess as sp

import pdb

class Trainer:
  def __init__(self,model,num_classes,args):
    self.args = args
    self.n_classes = num_classes
    self.model = model
    self.optimizer = Adam(model.parameters(), lr=args.learning_rate)
    self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    self.enable_gradient_clipping()
    self.cuda = to_cuda(args.gpu)
    self.writer = None
    self.scheduler = None

    if args.model_save_dir is not None:
        self.writer = SummaryWriter(os.path.join(args.model_save_dir, "logs"))
    if args.scheduler:
        self.scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)


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


  def compute_loss(self,pred_w,gold_w,debug=0):
    total_loss = []
    batch_size = gold_w.shape[0]

    mask = (gold_w!=PAD_ID).float() # [bs,W]
    sum_mask = mask.sum(1)
    sum_mask[sum_mask==0] = 1
    gold_w = self.cuda(fixed_var(gold_w.view(-1))) # [bs*W], pred_w: [bs*w,n_classes]
    loss = self.loss_function(pred_w,gold_w)       # [bs*W]
    loss = ((loss.view(batch_size,-1)*mask).sum(1) / sum_mask).sum()  # [1]
    
    return loss

      
  def train_batch(self, batch, gold_output, debug=0):
    """Train on one batch of sentences """
    self.model.train()
    batch_size = gold_output[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    total_loss = 0

    for w_seq,gold_w in zip(batch,gold_output):
      hidden = self.repackage_hidden(hidden) # ([]
      self.optimizer.zero_grad()
      pred_w,hidden = self.model.forward(w_seq, hidden)
      loss = self.compute_loss(pred_w, gold_w, debug)
      loss.backward()
      self.optimizer.step()
      total_loss += loss

    return total_loss.data


  def eval_batch(self,batch,gold_output,debug=0):
    self.model.eval()
    batch_size = gold_output[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    tloss = 0
    for w_seq,gold_w in zip(batch,gold_output):
      hidden = self.repackage_hidden(hidden) # ([]
      output,hidden = self.model.forward(w_seq, hidden)
      loss = self.compute_loss(output, gold_w, debug)
      tloss += loss
    return tloss.data


  def predict_batch(self,batch):
    self.model.eval()
    batch_size = batch[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    preds = []
    for w in batch:
      hidden = self.repackage_hidden(hidden) # ([]
      pred,hidden = self.model.predict(w,hidden)
      preds.append(pred)
    return preds


  def eval_metrics_batch(self,batch,data_vocabs,split='train',max_data=-1):
    """ eval lemmatizer using official script """
    cnt = 0
    stop_id = data_vocabs.vocab_oplabel.get_label_id(STOP_LABEL)
    forms_to_dump = []
    pred_lem_to_dump = []
    gold_lem_to_dump = []

    for op_seqs,forms,lemmas in batch.get_eval_batch():
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
          form_str = forms[i][j]
          if sum(w_op_seq)==0:
            pred_lemmas.append(form_str.lower())
            continue
            
          if stop_id in w_op_seq:
            _id = np.where(np.array(w_op_seq)==stop_id)[0][0]
            w_op_seq = w_op_seq[:_id+1]
          optokens = [data_vocabs.vocab_oplabel.get_label_name(x) \
                        for x in w_op_seq if x!=PAD_ID]
          
          pred_lemmas.append( apply_operations(form_str,optokens) )
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

    pobj = sp.run(["python3","2019/evaluation/evaluate_2019_task2.py",
                   "--reference", filename + ".conllu.gold",
                   "--output"   , filename + ".conllu.pred",
                  ], capture_output=True)

    try:
      output_res = pobj.stdout.decode().strip("\n").strip(" ").split("\t")
    except:
      print("-->Wrong eval output format")
      pdb.set_trace()
    output_res = [float(x) for x in output_res]

    return output_res[:2]


  def save_model(self,epoch):
    if self.args.model_save_dir is not None:
      if not os.path.exists(self.args.model_save_dir):
        os.makedirs(self.args.model_save_dir)
      model_save_file = os.path.join(self.args.model_save_dir, "{}_{}.pth".format("segm", epoch))
      print("Saving model to", model_save_file)
      torch.save(self.model.state_dict(), model_save_file)


  def update_summary(self,step,train_loss,dev_loss=None):
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

      self.writer.add_scalar("loss/loss_train", train_loss, step)
      if isinstance(dev_loss, torch.Tensor):
        self.writer.add_scalar("loss/loss_dev", dev_loss, step)
    #



