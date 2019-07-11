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
      gold_seq_lprob = gold_seq_lprob.squeeze().detach().cpu().numpy()

      op_weights = pred_w0.view(batch_size,-1).div(self.args.temperature).exp()
      lprob = F.log_softmax(op_weights,1)
      curr_tok = torch.multinomial(op_weights, s_size) # [bs,1]
      seq_log_prob = lprob.gather(1,curr_tok)

      curr_tok = curr_tok.view(-1,1)
      seq_log_prob = seq_log_prob.view(-1,1)
      tiled_pred_ids = [curr_tok.cpu().numpy()]
      tiled_hidden = self.repeat_hidden(hidden,s_size)
      mask = (curr_tok!=self.stop_id)

      for i in range(self.args.max_ops-1):
        logits,tiled_hidden = self.model.forward(curr_tok,tiled_hidden)
        op_weights = logits.div(self.args.temperature).exp()
        curr_tok = torch.multinomial(op_weights, 1).view(-1,1) # [bs,1]
        lprob = F.log_softmax(op_weights,1).gather(1,curr_tok) # get prob of sampled ops
        lprob *= mask.type(torch.float32)
        seq_log_prob += lprob
        mask *= (curr_tok!=self.stop_id)
        tiled_pred_ids.append(curr_tok.cpu().numpy())
      #
      tiled_pred_ids = np.hstack(tiled_pred_ids)
      seq_log_prob = seq_log_prob.view(-1).cpu().numpy()
      stop_mask = (tiled_pred_ids==self.stop_id).cumsum(1)==0
      sample_set_sum_lprob = self.cuda(torch.zeros([batch_size,1],dtype=torch.float32)).detach() # log sum_{s in S} p(s|...)

      # account for duplicate sampled sequences, keep the highest prob
      for i in range(batch_size):
        samples = defaultdict(lambda: -1000000.0)
        for j in range(i*s_size,(i+1)*s_size):
          try:
            smp = tuple([x for x in tiled_pred_ids[j, stop_mask[j,:]]])
          except:
            pdb.set_trace()
          samples[smp] = max(samples[smp],seq_log_prob[j])
        # get log_probs of samples in set
        s_lprobs = [x * self.args.alpha_q for x in samples.values()] + [self.args.alpha_q*gold_seq_lprob[i]]
        s_lprobs = self.cuda(torch.FloatTensor(s_lprobs)).detach()
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

      w0 = [self.loader.vocab_oplabel.get_label_name(x) for x in input_w0 if x!=self.pad_id]
      w0 = [get_action_components(x).segment for x in w0]
      delta = self.cuda(torch.zeros([batch_size,1],dtype=torch.float32)).detach()

      for i in range(batch_size):
        p_op = [self.loader.vocab_oplabel.get_label_name(x) \
                          for x in pred_ids[i,stop_mask[i,:]] if x!=self.pad_id]
        g_op = [self.loader.vocab_oplabel.get_label_name(x) \
                          for x in gold_ids[i,pad_mask[i,:]]]
        pred_lem,p_nvalid_ops = apply_operations(w0[i],p_op,ignore_start=False)
        gold_lem,g_nvalid_ops = apply_operations(w0[i],g_op,ignore_start=False)

        if g_nvalid_ops!=len(g_op)-1: # doesn't count STOP op
          pdb.set_trace()
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

    loss = (log_q_distr.exp() * delta).sum()
    
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
        hidden = self.repackage_hidden(hidden) # ([]
        output,hidden = self.model.forward(w_seq, hidden)
        loss = self.compute_loss(output, gold_w, debug)
        tloss += loss.item()
    return tloss




  def eval_metrics_batch(self,batch,data_vocabs,split='train',max_data=-1,
                         covered=False, dump_ops=False,
                         output_name=None):
    """ eval lemmatizer using official script """
    cnt = 0
    stop_id = data_vocabs.vocab_oplabel.get_label_id(STOP_LABEL)
    self.stop_id = stop_id
    forms_to_dump = []
    pred_lem_to_dump = []
    gold_lem_to_dump = []
    ops_to_dump = []

    for op_seqs,forms,lemmas in batch.get_eval_batch():
      predicted = self.predict_batch(op_seqs)
      predicted = batch.restore_batch(predicted) # bs x [ SxW ]

      forms_to_dump.extend(forms)
      gold_lem_to_dump.extend(lemmas)
      # get op labels & apply oracle 
      for i,sent in enumerate(predicted):
        sent = predicted[i]
        op_sent = []
        pred_lemmas = []
        len_sent = len(forms[i]) # forms and lemmas are not sent-padded
        for j in range(len_sent):
          w_op_seq = sent[j]
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
            pred_lemmas.append(form_str)
            continue
          if stop_id in w_op_seq:
            _id = np.where(np.array(w_op_seq)==stop_id)[0][0]
            w_op_seq = w_op_seq[:_id+1]
          optokens = [data_vocabs.vocab_oplabel.get_label_name(x) \
                        for x in w_op_seq if x!=PAD_ID]
          pred_lem,n_valid_ops = apply_operations(form_str,optokens)
          # pred_lem = pred_lem.replace(SPACE_LABEL," ") # <-- this doesn't have any effect
          pred_lemmas.append(pred_lem)
          if dump_ops:
            op_sent.append(" ".join([x for x in optokens[:n_valid_ops] if not x.startswith("STOP") and not x.startswith("START")]) )
        #

        if len(pred_lemmas)==0:
          pdb.set_trace()
        pred_lem_to_dump.append(pred_lemmas)
        ops_to_dump.append(op_sent)
      #

      cnt += op_seqs[0].shape[0]
      if max_data!=-1 and cnt > max_data:
        break
    #
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
      filename += ".lem"
    
    #pdb.set_trace()

    ops_to_dump = ops_to_dump if dump_ops else None
    dump_conllu(filename + ".conllu.gold",forms=forms_to_dump,lemmas=gold_lem_to_dump)
    dump_conllu(filename + ".conllu.pred",forms=forms_to_dump,lemmas=pred_lem_to_dump,ops=ops_to_dump)

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
                     train_acc=None,
                     dev_acc=None,
                     train_dist=None,
                     dev_dist=None
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

      for var,name in zip([train_loss,dev_loss,train_acc,dev_acc,train_dist,dev_dist],
                          ["loss/loss_train","loss/loss_dev",
                          "acc/train_acc","acc/dev_acc","dist/train_dist","dist/dev_dist"]):
        #if isinstance(dev_loss, torch.Tensor):
        if var != None:
          self.writer.add_scalar(name, var, step)
    #




