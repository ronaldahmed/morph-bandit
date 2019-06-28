from trainer_analizer_bundle import TrainerAnalizerBundle
from utils import MetricsWrap

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
    pred_seq = self.model.forward(batch,input_dec,hidden)
    total_loss = 0
    for dec_out,gold in zip(pred_seq,tgt_dec):
      total_loss += self.compute_loss(dec_out,gold)
    total_loss.backward()
    self.optimizer.step()
    
    return total_loss.item()