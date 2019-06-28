import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Module, Parameter, NLLLoss, LSTM, CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.functional import sigmoid, log_softmax, relu
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import to_cuda, fixed_var
from model_analizer import Analizer

import pdb


class AnalizerSeq(Analizer):
  def __init__(self,args,nvocab):
    super(AnalizerSeq, self).__init__(args,nvocab)
    

  def init_body(self,args,nvocab):
    # encoder: contexttualize actions w biLSTM / elmo / bert
    if args.op_aggr == "rnn":
      self.op_encoder = self.cuda(
        getattr(nn, args.rnn_type)(args.emb_size,args.op_enc_size,dropout=args.dropout,batch_first=True,bidirectional=True)
        )
    elif args.op_aggr=="cnn":
      self.op_encoder = None # not implemented yet

    # encoding at the word token level
    self.word_encoder = self.cuda(
        getattr(nn, args.rnn_type)(2*args.op_enc_size,args.w_enc_size,dropout=args.dropout,batch_first=True,bidirectional=True)
        )

    ## FEATURE DECODER
    self.max_length = 20
    self.dec_emb = self.cuda(nn.Embedding(self.nvocab, args.feat_dec_size))
    self.attn = self.cuda(nn.Linear(args.op_enc_size * 4 + args.feat_dec_size, self.max_length))
    self.attn_combine = self.cuda(nn.Linear(args.feat_dec_size * 2, args.feat_dec_size))
    self.dropout = nn.Dropout(args.dropout)
    self.decoder = self.cuda(
        getattr(nn, args.rnn_type)(args.op_enc_size,args.feat_enc_size,dropout=args.dropout,batch_first=True,bidirectional=False)
        )
    
    self.out = self.cuda(nn.Linear(args.feat_dec_size, nvocab))

    
    self.rnn_hidden = None
    #self.logprob = torch.nn.LogSoftmax()
    self.init_weights(nvocab)
    self.init_hidden(args.batch_size)


  def load_embeddings(self,):
    emb = self.cuda(nn.Embedding.from_pretrained(torch.load(self.args.embedding_pth).contiguous()) )
    for param in emb.parameters():
      param.requires_grad = False
    return emb


  def init_weights(self,nvocab):
    out_range = 1.0 / np.sqrt(2*self.args.feat_dec_size)
    
    self.out.bias.data.zero_()
    self.out.weight.data.uniform_(-out_range, out_range)
    


  """
  Hierarchical (two levels) encoder
  1. biLSTM to encode actions
  2. [fw,bw] of each action -> w_repr
  3. biLSTM over w_repr to obtain h_w (introduces context into w repr)
  4. h_action <- [op_fw;op_bw;hw_fw;hw_bw]
  """
  def encoder(self,batch, hidden):
    w_emb = []
    hidden_op,hidden_w = hidden
    batch_size = batch[0].shape[0]

    op_fw_bw_seq = []
    for w_op in batch:
      emb = self.emb(w_op)
      
      h_op,hidden_op  = self.op_encoder(emb,hidden_op) # h_op: [W,bs,2*size]
      h_op = h_op.view(batch_size,-1,2,self.args.op_enc_size)

      if self.args.op_repr == 'fw_bw':
        op_fw = h_op[:,:,0,:].view(batch_size,-1,self.args.op_enc_size)
        op_bw = h_op[:,:,1,:].view(batch_size,-1,self.args.op_enc_size)
        # append fw,bw for each time step
        op_fw_bw = torch.cat([op_fw,op_bw],2)
        op_fw_bw_seq.append(op_fw_bw)

        fw = h_op[:,-1,0,:].view(batch_size,1,self.args.op_enc_size)
        bw = h_op[:,0,1,:].view(batch_size,1,self.args.op_enc_size)
        fw_bw = torch.cat([fw,bw],2) # on rnn_size axis --> [bs,1,2*size]
        w_emb.append(fw_bw)

      #elif self.args.op_repr == 'self_att':      
    #
    w_seq = torch.cat(w_emb,1) # [bs,S,2*size]
    h_w, hidden_w = self.word_encoder(w_seq,hidden_w) # h_w: [S,bs,2*size]
    h_w = h_w.view(batch_size,-1,2,self.args.w_enc_size)

    hidw_fw = hidden_w[:,:,0,:].view(batch_size,-1,self.args.op_enc_size)
    hidw_bw = hidden_w[:,:,1,:].view(batch_size,-1,self.args.op_enc_size)
    encoder_hid_st = torch.cat([hidw_fw,hidw_bw],2)

    ctx_seqs = []

    # construct repr seq for each wop
    for i,op_fb in enumerate(op_fw_bw_seq):
      n_ops = op_fb.shape[1]
      h_tok_fw = h_w[:,i,0,:].view(batch_size,1,-1).repeat(1,n_ops,1)
      h_tok_bw = h_w[:,i,1,:].view(batch_size,1,-1).repeat(1,n_ops,1)
      # seq of actions w0,a1,a2: encoded seq
      ctx_action_seq = torch.cat([op_fb,h_tok_fw,h_tok_bw],2) # [fw,bw,w_fw,w_bw]

      yield ctx_action_seq,encoder_hid_st


  def attn_decoder(self,input,hidden,encoder_outputs):

    pdb.set_trace()
    
    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)

    attn_weights = F.softmax(
        self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                             encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.decoder(output, hidden)

    # output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights


  def forward(self, batch, dec_input, hidden):
    """ batch: Sx[bs x W] """
    dec_outputs = []
    k = 0
    for op_seq,enc_hid in self.encoder(batch,hidden):
      output,hid,attn_w = self.attn_decoder(dec_input[k],enc_hid,op_seq)
      dec_outputs.append(output)

    #

    pdb.set_trace()
    
    return dec_outputs


  def predict(self,batch,hidden):
    bs = batch[0].shape[0]
    output,hidden = self.forward(batch,hidden)
    pred = self.argmax(output).view(bs,-1).data.cpu().numpy()
    return pred,hidden


  def argmax(self,output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am


  def init_hidden(self, bsz):
    weight = next(self.parameters())
    if self.args.rnn_type == 'LSTM':
      self.rnn_hidden = [(self.cuda(weight.new_zeros(2,bsz,self.args.op_enc_size)),
                          self.cuda(weight.new_zeros(2,bsz,self.args.op_enc_size)) ),
                         (self.cuda(weight.new_zeros(2,bsz,self.args.w_enc_size)),
                          self.cuda(weight.new_zeros(2,bsz,self.args.w_enc_size)) )]
                         
    else:
      self.rnn_hidden = [self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size)),
                         self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size)) ]
                        

  def repackage_tensors(self,h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
      return tuple(self.repackage_tensors(v) for v in h)

  def flush_tensors(self,h):
    if isinstance(h, torch.Tensor):
      return h.data.zero_()
    else:
      return tuple(self.flush_tensors(v) for v in h)

  def make_continuous(self,h):
    if isinstance(h, torch.Tensor):
      return h.contiguous()
    else:
      return tuple(self.make_continuous(v) for v in h)


  def slice(self,h,bs):
    if isinstance(h, torch.Tensor):
      return h[:,:bs,:]
    else:
      return tuple(self.slice(v,bs) for v in h)


  def refactor_hidden(self,batch_size):
    hidden = self.flush_tensors(self.rnn_hidden)
    hidden = self.slice(hidden,batch_size)
    hidden = self.make_continuous(hidden)
    hidden = self.repackage_tensors(hidden)
    return hidden