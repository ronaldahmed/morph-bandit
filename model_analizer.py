from tensorboardX import SummaryWriter
import torch
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Module, Parameter, NLLLoss, LSTM, CrossEntropyLoss
from torch.nn.functional import sigmoid, log_softmax, relu
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import to_cuda, fixed_var
import numpy as np
import pdb


class Analizer(Module):
  def __init__(self,args,nvocab):
    super(Lemmatizer, self).__init__()
    self.args = args
    self.cuda = to_cuda(args.gpu)
    self.drop = nn.Dropout(args.dropout)
    self.emb = self.load_embeddings()
    self.op_encoder = ""

    if args.op_aggr == "rnn":
      self.op_encoder = self.cuda(
        getattr(nn, args.rnn_type)(args.emb_size,args.rnn_size,dropout=args.dropout,batch_first=True,bidirectional=True)
        )
    elif args.op_aggr=="cnn":
      self.op_encoder = None # not implemented yet

    self.word_encoder = self.cuda(
        getattr(nn, args.rnn_type)(2*args.rnn_size,args.rnn_size,dropout=args.dropout,batch_first=True,bidirectional=True)
        )

    self.ff1 = self.cuda(nn.Linear(2*args.rnn_size,args.mlp_size))
    self.ff2 = self.cuda(nn.Linear(args.mlp_size, nvocab))

    self.rnn_hidden = None
    #self.logprob = torch.nn.LogSoftmax()
    self.init_weights(nvocab)
    self.init_hidden(args.batch_size)


  def load_embeddings(self,):
    return self.cuda(fixed_var(torch.load(self.args.embedding_pth)))


  def init_weights(self,nvocab):
    ff1_range = 1.0 / np.sqrt(self.args.rnn_size)
    ff2_range = 1.0 / np.sqrt(self.args.mlp_size)
    self.emb.weight.data.uniform_(-emb_range, emb_range)
    self.ff1.bias.data.zero_()
    self.ff1.weight.data.uniform_(-ff1_range, ff1_range)
    self.ff2.bias.data.zero_()
    self.ff2.weight.data.uniform_(-ff2_range, ff2_range)


  def forward(self, batch, hidden):
    """ batch: Sx[bs x W] """
    w_emb = []
    hidden_op,hidden_w = hidden
    batch_size = batch[0].shape[0]
    for w_op in batch:
      emb = self.emb(w_op)
      h_op,hidden_op  = self.op_encoder(emb,hidden_op) # h_op: [W,bs,2*size]
      h_op = h_op.view(-1,batch_size,2,self.args.rnn_size)
      fw_bw = [ h_op[-1,:,0,,:].view(1,batch_size,self.args.rnn_size),
                h_op[0,:,1,:].view(1,batch_size,self.args.rnn_size)]
      fw_bw = torch.cat(fw_bw,2) # on rnn_size axis --> [1,bs,2*size]
      w_emb.append(fw_bw)
    #
    w_seq = torch.cat(w_emb,0) # [S,bs,2*size]
    h_w, hidden_w = self.word_encoder(w_seq,hidden_w) # h_w: [S,bs,2*size]
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
      self.rnn_hidden = [(self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size)),
                          self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size))),
                         (self.cuda(weight.new_zeros(2,bsz,2*self.args.rnn_size)),
                          self.cuda(weight.new_zeros(2,bsz,2*self.args.rnn_size))) ]
                         
    else:
      self.rnn_hidden = [self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size)),
                         self.cuda(weight.new_zeros(2,bsz,2*self.args.rnn_size)) ]
                        
