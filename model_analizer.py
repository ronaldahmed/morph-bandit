from tensorboardX import SummaryWriter
import torch
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
from torch.nn import Module, Parameter, NLLLoss, LSTM, CrossEntropyLoss
from torch.nn.functional import sigmoid, log_softmax
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import to_cuda
import pdb


class Analizer(Module):
  def __init__(self,args,nvocab):
    super(Analizer, self).__init__()
    self.args = args
    self.cuda = to_cuda(args.gpu)
    self.drop = torch.nn.Dropout(args.dropout)
    self.emb = self.cuda(torch.nn.Embedding(nvocab, args.emb_size))
    self.encoder = self.cuda(
      getattr(torch.nn, args.rnn_type)(args.emb_size,args.rnn_size,dropout=args.dropout,batch_first=True)
      )
    self.mlp = self.cuda(torch.nn.Linear(args.mlp_size, nvocab))
    self.rnn_hidden = None
    #self.logprob = torch.nn.LogSoftmax()
    self.init_weights()
    self.init_hidden(args.batch_size)


  def init_weights(self):
    initrange = 0.1
    self.emb.weight.data.uniform_(-initrange, initrange)
    self.mlp.bias.data.zero_()
    self.mlp.weight.data.uniform_(-initrange, initrange)


  def forward(self, w, hidden):
    emb = self.emb(w)
    rnn_output, hidden = self.encoder(emb, hidden)
    rnn_output = self.drop(rnn_output).contiguous()
    output_flat = self.mlp(
              rnn_output.view(
                  rnn_output.size(0) * rnn_output.size(1), 
                  rnn_output.size(2)
                  )
              )  
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
