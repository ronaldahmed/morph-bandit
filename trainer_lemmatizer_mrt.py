import torch
import numpy as np
import torch.nn.functional as F
from trainer_lemmatizer_mle import TrainerLemmatizerMLE
from collections import defaultdict

from utils import apply_operations, \
                  get_action_components, \
                  PAD_ID, \
                  STOP_LABEL, \
                  SPACE_LABEL

from evaluate_2019_task2 import distance

import pdb

EPS = 1e-32

class TrainerLemmatizerMRT(TrainerLemmatizerMLE):
  def __init__(self,model,loader,args):
    super(TrainerLemmatizerMRT,self).__init__(model,loader,args)
  
  def init_loss_objective(self,):
    self.loss_function = None

    
  def sample_action_space(self,pred_w0,hidden,gold_seq_lprob):
    """
    Get sampled space S(a) for posterior approximation
    one S per w_0, batch_size S sets in total
    - pred_w0: (w0) [bs x op_vocab]
    - gold_w: (a1 a2 STOP) [bs x max_actions]
    - hidden: (c,st)
    - gold_seq_lprob: log_prob of gold action sequence [bs x 1]
    """
    assert self.stop_id != -1
    batch_size = pred_w0.shape[0]
    s_size = self.args.sample_space_size
    
    # get prob of K sampled seqs
    with torch.no_grad():
      gold_seq_lprob = gold_seq_lprob.squeeze()#.detach().cpu().numpy()

      op_weights = pred_w0.view(batch_size,-1).div(self.args.temperature).exp()
      lprob = F.log_softmax(op_weights,1)
      curr_tok = torch.multinomial(op_weights, s_size).detach() # [bs,1]
      seq_log_prob = lprob.gather(1,curr_tok)

      curr_tok = curr_tok.view(-1,1)
      seq_log_prob = seq_log_prob.view(-1,1)
      tiled_pred_ids = [curr_tok.detach().cpu().numpy()]
      tiled_hidden = self.repeat_hidden(hidden,s_size)
      mask = (curr_tok!=self.stop_id)

      for i in range(self.args.max_ops-1):
        logits,tiled_hidden = self.model.forward(curr_tok,tiled_hidden)
        op_weights = logits.div(self.args.temperature).exp()
        curr_tok = torch.multinomial(op_weights, 1).view(-1,1).detach() # [bs,1]
        lprob = F.log_softmax(op_weights,1).gather(1,curr_tok) # get prob of sampled ops
        lprob *= mask.type(torch.float32)
        seq_log_prob += lprob
        mask *= (curr_tok!=self.stop_id)
        tiled_pred_ids.append(curr_tok.detach().cpu().numpy())
      #
      tiled_pred_ids = np.hstack(tiled_pred_ids)
      seq_log_prob = seq_log_prob.view(-1).cpu().numpy()
      stop_mask = (tiled_pred_ids==self.stop_id).cumsum(1)==0
      sample_set_sum_lprob = self.cuda(torch.zeros([batch_size,1],dtype=torch.float32)).detach() # log sum_{s in S} p(s|...)

      # account for duplicate sampled sequences, keep the highest prob
      for i in range(batch_size):
        samples = {}
        s_lprobs = self.args.alpha_q*gold_seq_lprob[i]
        for j in range(i*s_size,(i+1)*s_size):
          smp = tuple([x for x in tiled_pred_ids[j, stop_mask[j,:]]])
          if smp in samples: continue
          s_lprobs += self.args.alpha_q * seq_log_prob[j]
        # get log_probs of samples in set
        # s_lprobs = self.cuda(torch.FloatTensor(s_lprobs)).detach()
        sample_set_sum_lprob[i,0] = torch.logsumexp(s_lprobs,0)
      #
    #
    return sample_set_sum_lprob


  def normalized_distance(self,str1,str2):
    """ from 0:max(len_s1,len_s2) -> [-1,1] """
    return (2.0 * distance(str1,str2) / max(len(str1),len(str2))) - 1.0


  def calc_delta(self,input_w0,pred_ids,gold_ids,):
    """
    Calculate loss function Delta = f(accuracy,edit_distance)
    - input_w0: (w0) [bs x 1]
    - pred_ids: predicted action ids [bs x max_ops]
    - gold_ids: gold/true action ids [bs x max_ops]
    """
    batch_size = input_w0.shape[0]
    with torch.no_grad():
      # input_w0 = input_w0.view(-1).cpu().numpy()
      # pred_ids = pred_ids.cpu().numpy()
      # gold_ids = gold_ids.cpu().numpy()

      stop_mask = (pred_ids==self.stop_id).cumsum(1)==0
      pad_mask = (gold_ids==self.pad_id).cumsum(1)==0

      w0 = [self.loader.vocab_oplabel.get_label_name(x) for x in input_w0]
      w0 = [get_action_components(x).segment for x in w0]
      delta = self.cuda(torch.zeros([batch_size,1],dtype=torch.float32)).detach()

      for i in range(batch_size):
        if w0[i] is None: continue  # PAD gets None
        p_op = [self.loader.vocab_oplabel.get_label_name(x) \
                          for x in pred_ids[i,stop_mask[i,:]] if x!=self.pad_id]
        g_op = [self.loader.vocab_oplabel.get_label_name(x) \
                          for x in gold_ids[i,pad_mask[i,:]]]
        pred_lem,p_nvalid_ops = apply_operations(w0[i],p_op,ignore_start=False)
        gold_lem,g_nvalid_ops = apply_operations(w0[i],g_op,ignore_start=False)

        # if g_nvalid_ops!=len(g_op)-1: # doesn't count STOP op
        #   pdb.set_trace()
        delta[i,0] = self.normalized_distance(pred_lem,gold_lem) - float(pred_lem==gold_lem)
      #
      # delta = torch.log(delta)
      if any(torch.isnan(delta)):
        pdb.set_trace()
        print("-->")

    return delta


  def compute_loss(self,input_w0,pred_w,gold_w,hidden):
    """ Implements Minimum Risk Loss
    """
    batch_size,max_ops = pred_w.shape[:2]
    # get lprob of gold sequence in batch
    pred_logp = F.log_softmax(pred_w,2)
    pred_seq_lprob,pred_seq_ids = pred_logp.max(2) # [bs x max_ops]
    stop_mask = pred_seq_ids==self.stop_id
    stop_pos = max_ops - 1 - stop_mask.flip([1]).argmax(1)
    stop_prob = (stop_mask.sum(1,keepdim=True)>0).type(torch.float32) * pred_seq_lprob.gather(1,stop_pos.view(-1,1))
    
    stop_mask = (stop_mask.cumsum(1)==0).type(torch.float32) # 111(valid)000(after stop)
    pred_seq_lprob = (pred_seq_lprob*stop_mask).sum(1).view(-1,1)
    pred_seq_lprob += stop_prob

    gold_seq_lprob = pred_logp.gather(2,gold_w.view(batch_size,-1,1)).squeeze()
    pad_mask = ((gold_w==self.pad_id).cumsum(1)==0).type(torch.float32)
    gold_seq_lprob = (gold_seq_lprob*pad_mask).sum(1).view(-1,1)

    # sum of p(a'|...) over sampled action 
    sample_set_lprob = self.sample_action_space(pred_w[:,0,:].squeeze(),hidden,gold_seq_lprob)
    # log(Delta(y,y^))
    delta = self.calc_delta(input_w0,pred_seq_ids,gold_w)
    # log(Q(y|...))
    log_q_distr = self.args.alpha_q * pred_seq_lprob - sample_set_lprob # the propag anchor is lp(gold)

    loss = (log_q_distr.exp() * delta).mean()
    
    if torch.isnan(loss): pdb.set_trace()

    return loss

      
  def train_batch(self, batch, gold_output, debug=0):
    """Train on one batch of sentences """
    self.model.train()

    batch_size = gold_output[0].shape[0]
    #batch_size = len(gold_output[0])
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,0,batch_size)
    total_loss = 0
    for w_seq,gold_w in zip(batch,gold_output):
      nops = w_seq.shape[1]
      hidden = self.repackage_hidden(hidden) # ([]
      self.optimizer.zero_grad()
      # un pasito pa qui
      pred_w0,hidden0 = self.model.forward(w_seq[:,0].view(-1,1), hidden)
      pred_w0 = pred_w0.view(batch_size,-1,self.n_classes)
      # un pasito pa lla
      pred_w = []
      if nops==1:
        pred_w = pred_w0
        hidden = hidden0
      else:
        pred_w1n,hidden = self.model.forward(w_seq[:,1:].view(batch_size,-1), hidden0)
        pred_w1n = pred_w1n.view(batch_size,-1,self.n_classes)
        pred_w = torch.cat([pred_w0,pred_w1n],1)
      loss = self.compute_loss(w_seq[:,0].view(-1,1), pred_w, gold_w, hidden0)
      loss.backward()
      self.optimizer.step()
      total_loss += loss.item()

    return total_loss


  def eval_batch(self,batch,gold_output,debug=0):
    self.model.eval()
    batch_size = gold_output[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,0,batch_size)
    tloss = 0
    with torch.no_grad():
      for w_seq,gold_w in zip(batch,gold_output):
        nops = w_seq.shape[1]
        hidden = self.repackage_hidden(hidden) # ([]
        pred_w0,hidden0 = self.model.forward(w_seq[:,0].view(-1,1), hidden)
        pred_w0 = pred_w0.view(batch_size,-1,self.n_classes)
        # un pasito pa lla
        pred_w = []
        if nops==1:
          pred_w = pred_w0
          hidden = hidden0
        else:
          pred_w1n,hidden = self.model.forward(w_seq[:,1:].view(batch_size,-1), hidden0)
          pred_w1n = pred_w1n.view(batch_size,-1,self.n_classes)
          pred_w = torch.cat([pred_w0,pred_w1n],1)
        loss = self.compute_loss(w_seq[:,0].view(-1,1), pred_w, gold_w, hidden0)
        tloss += loss.item()
    return tloss

