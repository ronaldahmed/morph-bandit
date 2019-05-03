import sys
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from time import monotonic
from utils import to_cuda, fixed_var
from data_utils import PAD_ID
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
        writer = SummaryWriter(os.path.join(model_save_dir, "logs"))
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


  def compute_loss(self,pred,gold_output,debug=0):
    time1 = monotonic()
    total_loss = []
    batch_size = gold_output[0].shape[0]
    for pred_w,gold_w in zip(pred,gold_output):
      mask = (gold_w!=PAD_ID).float() # [bs,W]
      gold_w = self.cuda(fixed_var(gold_w.view(-1))) # [bs*W], pred_w: [bs*w,n_classes]
      loss = self.loss_function(pred_w,gold_w)       # [bs*W]
      loss = ((loss.view(batch_size,-1)*mask).sum(1) / mask.sum(1) ).sum() # [1]
      total_loss.append(loss)
    total_loss = sum(total_loss)
    if debug:
      time2 = monotonic()
      print("Forward total in loss: {}".format(round(time2 - time1, 3)))
    return total_loss

      
  def train_batch(self, batch, gold_output, debug=0):
    """Train on one batch of sentences """
    self.model.train()
    batch_size = gold_output[0].shape[0]
    hidden = self.model.init_hidden(batch_size)
    hidden = self.repackage_hidden(hidden)
    output = self.model.forward(batch, hidden)
    self.optimizer.zero_grad()
    time0 = monotonic()
    loss = self.compute_loss(output, gold_output, debug)
    time1 = monotonic()
    loss.backward()
    time2 = monotonic()

    self.optimizer.step()
    if debug:
      time3 = monotonic()
      print("Time in loss: {}, time in backward: {}, time in step: {}".format(round(time1 - time0, 3),
                                                                              round(time2 - time1, 3),
                                                                              round(time3 - time2, 3)))
    return loss.data


  def update_summary(self,epoch,train_loss,dev_loss):
    if self.writer is not None:
      for name, param in self.model.named_parameters():
        self.writer.add_scalar("parameter_mean/" + name,
                          param.data.mean(),
                          epoch)
        self.writer.add_scalar("parameter_std/" + name, param.data.std(), ep)
        if param.grad is not None:
            self.writer.add_scalar("gradient_mean/" + name,
                              param.grad.data.mean(),
                              epoch)
            self.writer.add_scalar("gradient_std/" + name,
                              param.grad.data.std(),
                              epoch)

      self.writer.add_scalar("loss/loss_train", train_loss, epoch)
      self.writer.add_scalar("loss/loss_dev", dev_loss, epoch)
    #


  


  def evaluate_batch(self,batch,data_obj):
    """ eval lemmatizer using official script """
    self.model.eval()
    for op_seqs,forms,lemmas in batch.get_eval_batch():
      predicted = self.model.predict(op_seqs)






