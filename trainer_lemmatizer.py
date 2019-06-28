import sys, os
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from torch.nn.functional import log_softmax
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

class BeamNode:
  def __init__(self,c_t,w_list,log_prob):
    self._c = c_t
    self._op_list = w_list
    self._lprob = log_prob
    self._finished = False


class TrainerLemmatizer:
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
    self.stop_id = -1
    self.pad_id = PAD_ID

    if args.model_save_dir is not None:
        self.writer = SummaryWriter(os.path.join(args.model_save_dir, "logs"))
    if args.scheduler:
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', 0.1, 10, True)


  def freeze_model(self):
    for param in self.model.parameters():
      param.requires_grad = False


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

  def slice(self,h,init=0,end=-1):
    if isinstance(h, torch.Tensor):
      return h[:,init:end,:]
    else:
      return tuple(self.slice(v,init,end) for v in h)

  def concat_hidden(self,h):
    if isinstance(h, torch.Tensor):
      return torch.cat(h,1)
    else:
      return tuple(torch.cat([x[i] for x in h] ,1) for i in range(len(h[0])))


  def compute_loss(self,pred_w,gold_w,debug=0):
    total_loss = []
    batch_size = gold_w.shape[0]

    mask = (gold_w!=PAD_ID).float() # [bs,W]
    sum_mask = mask.sum(1)
    sum_mask[sum_mask==0] = 1
    #gold_w = self.cuda(fixed_var(gold_w.view(-1))) # [bs*W], pred_w: [bs*w,n_classes]
    gold_w = gold_w.view(-1)
    loss = self.loss_function(pred_w,gold_w)       # [bs*W]
    loss = ((loss.view(batch_size,-1)*mask).sum(1) / sum_mask).sum()  # [1]
    
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
      hidden = self.repackage_hidden(hidden) # ([]
      self.optimizer.zero_grad()
      pred_w,hidden = self.model.forward(w_seq, hidden)
      loss = self.compute_loss(pred_w, gold_w, debug)
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

  # def predict_batch(self,batch):
  #   """ previous version, uses gold input to gerate 1-op shifted seq [WRONG]
  #   """
  #   self.model.eval()
  #   batch_size = batch[0].shape[0]
  #   hidden = self.flush_hidden(self.model.rnn_hidden)
  #   hidden = self.slice(hidden,0,batch_size) #monotonic
  #   preds = []
  #   with torch.no_grad():
  #     for w in batch:
  #       hidden = self.repackage_hidden(hidden) # ([]
  #       pred,hidden = self.model.predict(w,hidden)
  #       preds.append(pred)
  #   return preds


  def predict_batch(self,batch,start=False,score=False):
    """ redirects to decoding strategy implementations
    """
    if self.args.beam_size == -1:
      return self.greedy_decoder(batch,start,score)
    else:
      return self.beam_search_decoder(batch,self.args.beam_size,start=start)



  def greedy_decoder(self,batch,start=False,score=False):
    """ Start with initial form and sample from LM until 
        reaching STOP or MAX_OPS
    """
    self.model.eval()
    batch_size = batch[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    hidden = self.slice(hidden,0,batch_size)
    pred_batch = []
    pred_score = []
    with torch.no_grad():
      for w in batch:
        hidden = self.repackage_hidden(hidden) # ([]
        curr_tok = w
        pred_w = []
        pred_sc = []
        if start: pred_w.append(w)
        for i in range(self.args.max_ops):
          output,hidden = self.model.forward(curr_tok,hidden)
          logits = output.view(batch_size,-1)
          op_weights = logits.div(self.args.temperature).exp()
          # print("--> sum w weights: ",torch.sum(op_weights).data)
          # if torch.sum(op_weights).data < (1e-12)*op_weights.shape[0]*op_weights.shape[1]:
          #   print("-> found zeroes!!")
          #   pred_w.append(fixed_var( self.cuda(torch.LongTensor(np.zeros([batch_size,1]))) ) )
          #   break
          op_idx = torch.multinomial(op_weights, 1) # [bs,1]
          curr_tok = op_idx
          pred_w.append( op_idx )

          if score:
            opw = torch.nn.functional.softmax(logits,1).cpu().numpy() # [bs x n_ops]
            op_idx_ = op_idx.cpu().numpy()
            sm_distr = np.zeros([batch_size,1])
            for i in range(batch_size):
              sm_distr[i,0] = opw[i,op_idx_[i,0]]
            pred_sc.append(sm_distr)
          #
        #
        pred_w = torch.cat(pred_w,1)
        pred_w = pred_w.cpu().numpy()
        pred_batch.append(pred_w)
        if score:
          pred_sc = np.hstack(pred_sc)
          pred_score.append(pred_sc)
      #

    if score:
      return pred_batch,pred_score
    return pred_batch


  def relative_prunner(self,candidates):
    """ candidates must be rev-sorted by log_prob """
    thr = np.log(self.args.rel_prunning) + candidates[0]._lprob
    filtered = [x for x in candidates if x._lprob >= thr ]
    if len(filtered)==0:
      pdb.set_trace()

    return filtered


  def beam_search_decoder(self,sent_batch,beam_size=5,start=False):
    """ Implements beam search decoding
      sent_batch: Sx[bs x 1]
    """
    self.model.eval()
    batch_size = sent_batch[0].shape[0]
    hidden = self.flush_hidden(self.model.rnn_hidden)
    pred_batch = []
    cnt = 0
    with torch.no_grad():
      for w in sent_batch:
        hidden = self.repackage_hidden(hidden)
        hidden_next = []
        op_batch = []
        for i in range(batch_size):
          c_0 = self.slice(hidden,i,i+1)
          w_0 = w[i,:]
          node_0 = BeamNode(c_0,[w_0],0)
          A_prev = [node_0]
          A_next = []

          all_finished = False
          while not all_finished:
            all_finished = True
            n_to_exp = len([x for x in A_prev if not x._finished])
            if n_to_exp==0:
              A_next = A_prev
              break

            A_prev = [x for x in A_prev if not x._finished]
            # make a synthetic batch made out of all nodes
            w_t_1 = [node._op_list[-1].view(1,1) for node in A_prev]
            c_t_1 = [node._c for node in A_prev]
            
            all_finished = False
            c_t_1 = self.concat_hidden(c_t_1)
            w_t_1 = torch.cat(w_t_1,0)

            logit_t,c_t = self.model.forward(w_t_1,c_t_1)
            lpd_w_t = log_softmax(logit_t,1)
            scores,posts = torch.topk(lpd_w_t,beam_size,1)

            scores = scores.cpu().numpy()
            posts = posts.cpu().numpy()

            for j in range(n_to_exp):
              c_t_j = self.slice(c_t,j,j+1)
              node = A_prev[j]
              for k in range(beam_size):
                lp_w_t = scores[j,k]
                w_t = posts[j,k]
                op_list = node._op_list + [self.cuda(torch.LongTensor([w_t]))]
                cand_node = BeamNode(c_t_j,op_list,node._lprob + lp_w_t)
                if any([w_t==self.stop_id,
                        w_t==self.pad_id,
                        len(cand_node._op_list)>=self.args.max_ops]) :
                  cand_node._finished = True
                A_next.append(cand_node)
            #
            
            A_next.sort(reverse=True,key=lambda x:x._lprob)
            A_prev = A_next[:beam_size]
            if self.args.rel_prunning > 0.0:
              A_prev = self.relative_prunner(A_prev)
            A_next = []
            # print([len(x._op_list) for x in A_prev])
          #END-WHILE
          
          optm_op_seq = torch.cat(A_prev[0]._op_list,0).cpu().numpy().tolist()
          hidden_next.append(A_prev[0]._c)
          op_batch.append(optm_op_seq)
        #END-FOR-BATCH
        max_op_len = max([len(x) for x in op_batch])
        for j in range(batch_size):
          temp = np.array(op_batch[j] + [self.pad_id]*(max_op_len - len(op_batch[j])))
          op_batch[j] = np.reshape(temp,[1,-1])
        op_batch = np.vstack(op_batch)
        if not start:
          op_batch = op_batch[:,1:]

        hidden = self.concat_hidden(hidden_next)
        pred_batch.append(op_batch)

        if cnt % 10 == 0:
          print("->",cnt)
        cnt += 1
      #END-FOR-W_S
    #
    
    return pred_batch


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




