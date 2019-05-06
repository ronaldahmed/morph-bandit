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
from utils import to_cuda
import numpy as np
import pdb


class Analizer(Module):
  def __init__(self,args,nvocab):
    super(Analizer, self).__init__()
    self.args = args
    self.cuda = to_cuda(args.gpu)
    self.drop = nn.Dropout(args.dropout)
    self.emb = self.cuda(nn.Embedding(nvocab, args.emb_size))
    self.encoder = self.cuda(
      getattr(torch.nn, args.rnn_type)(args.emb_size,args.rnn_size,dropout=args.dropout,batch_first=True)
      )
    self.ff1 = self.cuda(nn.Linear(args.rnn_size,args.mlp_size))
    self.ff2 = self.cuda(nn.Linear(args.mlp_size, nvocab))

    self.rnn_hidden = None
    #self.logprob = torch.nn.LogSoftmax()
    self.init_weights(nvocab)
    self.init_hidden(args.batch_size)


  def init_weights(self,nvocab):
    emb_range = 1.0 / np.sqrt(nvocab)
    ff1_range = 1.0 / np.sqrt(self.args.rnn_size)
    ff2_range = 1.0 / np.sqrt(self.args.mlp_size)
    self.emb.weight.data.uniform_(-emb_range, emb_range)
    self.ff1.bias.data.zero_()
    self.ff1.weight.data.uniform_(-ff1_range, ff1_range)
    self.ff2.bias.data.zero_()
    self.ff2.weight.data.uniform_(-ff2_range, ff2_range)


  def forward(self, w, hidden):
    emb = self.emb(w)
    rnn_output, hidden = self.encoder(emb, hidden)
    rnn_output = self.drop(rnn_output) # only works when num_layers > 1
    rnn_shape = rnn_output.shape
    rnn_output = rnn_output.contiguous().view(rnn_shape[0]*rnn_shape[1],rnn_shape[2])
    ff1 = self.drop(relu(self.ff1(rnn_output)))
    output_flat = self.ff2(ff1)
    
    return output_flat,hidden


  def predict(self,w,hidden):
    preds = []
    bs = w.shape[0]
    output,hidden = self.forward(w,hidden)
    pred = self.argmax(output).view(bs,-1).data.cpu().numpy()
    return pred,hidden


  def argmax(self,output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am


  def init_hidden(self, bsz):
    weight = next(self.parameters())
    if self.args.rnn_type == 'LSTM':
      self.rnn_hidden = (self.cuda(weight.new_zeros(1,bsz,self.args.rnn_size)),
                         self.cuda(weight.new_zeros(1,bsz,self.args.rnn_size)))
    else:
      self.rnn_hidden = self.cuda(weight.new_zeros(1,bsz,self.args.rnn_size))
