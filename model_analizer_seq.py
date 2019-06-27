import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Module, Parameter, NLLLoss, LSTM, CrossEntropyLoss
from torch.nn.functional import sigmoid, log_softmax, relu
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import to_cuda, fixed_var
from model_analyzer import Analizer

import pdb


class AnalizerSeq(Analizer):
  def __init__(self,args,nvocab):
    super(AnalizerSeq, self).__init__()
    self.args = args
    self.cuda = to_cuda(args.gpu)
    self.drop = nn.Dropout(args.dropout)
    self.emb = self.load_embeddings()
    self.op_encoder = ""

    # encoder: contexttualize actions w biLSTM / elmo / bert
    if args.op_aggr == "rnn":
      self.op_encoder = self.cuda(
        getattr(nn, args.rnn_type)(args.emb_size,args.op_enc_size,dropout=args.dropout,batch_first=True,bidirectional=True)
        )
    
    # encoding at the word token level
    self.word_encoder = self.cuda(
        getattr(nn, args.rnn_type)(2*args.op_enc_size,args.w_enc_size,dropout=args.dropout,batch_first=True,bidirectional=True)
        )

    # decode features for each token
    #   aggregate fwd & bwd
    self.decoder = self.cuda(
        getattr(nn, args.rnn_type)(args.op_enc_size,args.feat_enc_size,dropout=args.dropout,batch_first=True,bidirectional=False)
        )

    self.ff1 = self.cuda(nn.Linear(args.feat_enc_size,args.feat_mlp_size))
    self.ff2 = self.cuda(nn.Linear(args.feat_mlp_size, nvocab))

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
    ff1_range = 1.0 / np.sqrt(2*self.args.w_enc_size)
    ff2_range = 1.0 / np.sqrt(self.args.w_mlp_size)
    self.ff1.bias.data.zero_()
    self.ff1.weight.data.uniform_(-ff1_range, ff1_range)
    self.ff2.bias.data.zero_()
    self.ff2.weight.data.uniform_(-ff2_range, ff2_range)


  def forward(self, batch, hidden):
    """ batch: Sx[bs x W] """
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
        op_fw_bw = torch.cat([op_fw,op_bw],1)
        op_fw_bw_seq.append(op_fw_bw)

        fw = h_op[:,-1,0,:].view(batch_size,1,self.args.op_enc_size)
        bw = h_op[:,0,1,:].view(batch_size,1,self.args.op_enc_size)
        fw_bw = torch.cat(fw_bw,2) # on rnn_size axis --> [bs,1,2*size]
        w_emb.append(fw_bw)

      #elif self.args.op_repr == 'self_att':      
    #
    w_seq = torch.cat(w_emb,1) # [bs,S,2*size]
    h_w, hidden_w = self.word_encoder(w_seq,hidden_w) # h_w: [S,bs,2*size]
    h_w = h_w.view(batch_size,-1,2,self.args.w_enc_size)

    # construct repr seq for each wop
    for i,op_fb in enumerate(op_fw_bw_seq):
      h_tok_fw = h_w[:,i,0,:].view(batch_size,1,-1)
      h_tok_bw = h_w[:,i,1].view(batch_size,1,-1)
      
      # add [hw_f,fw_bw] to each op fw_bw

      action_repr = torch.cat(op_fb)


    rnn_shape = h_w.shape
    sent_output = h_w.contiguous().view(rnn_shape[0]*rnn_shape[1],rnn_shape[2])
    ff1 = self.drop(relu(self.ff1(sent_output)))
    output_flat = self.ff2(ff1)
    
    return output_flat,[hidden_op,hidden_w]


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