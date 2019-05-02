from tensorboardX import SummaryWriter
import torch
from torch import FloatTensor, LongTensor, cat, mm, norm, randn, zeros, ones
from torch.autograd import Variable
from torch.nn import Module, Parameter, NLLLoss, LSTM, CrossEntropyLoss
from torch.nn.functional import sigmoid, log_softmax
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Analizer(Module):
  def __init__(self,args,nvocab):
    super(Analizer, self).__init__()
    self.args = args
    self.to_cuda = to_cuda(args.gpu)
    self.drop = nn.Dropout(dropout)
    self.criterion = CrossEntropyLoss()
    
    self.emb = nn.Embedding(nvocab, args.emb_size)
    self.encoder = LSTM(args.emb_size,args.rnn_size,dropout=args.dropout,batch_first=True)
    self.mlp = nn.Linear(args.mlp_size, nvocab)
    self.logprob = nn.LogSoftmax()
    self.init_weights()


  def init_weights(self):
    initrange = 0.1
    self.emb.weight.data.uniform_(-initrange, initrange)
    self.encoder.bias.data.zero_()
    self.encoder.weight.data.uniform_(-initrange, initrange)


  def forward(self, input, hidden):
    # emb = self.drop(self.emb(input))
    # output = self.mlp(rnn_output)
    output = []
    for w in input:
      emb = self.emb(w)
      rnn_output, hidden = self.encoder(emb, hidden)
      rnn_output = self.drop(rnn_output)

      output_flat = self.mlp(
                rnn_output.view(
                    rnn_output.size(0) * rnn_output.size(1), 
                    rnn_output.size(2)
                    )
                )
      output.append(output_flat)

    return output


  def predict(self, input):
    # emb = self.drop(self.emb(input))
    # output = self.mlp(rnn_output)
    hidden = (torch.zeros(args.batch_size,1, self.args.rnn_size),
              torch.zeros(args.batch_size,1, self.args.rnn_size))
    output = self.forward(input,hidden)
    preds = []
    # wseq_len = output[0].size(0) // args.batch_size
    for w_seq in output:
      pred = argmax(w_seq).view(self.args.batch_size,-1).data.numpy()
      preds.append(pred)
    return pred


  def argmax(output):
    """ only works for kxn tensors """
    _, am = torch.max(output, 1)
    return am

  def init_hidden(self, bsz):
    weight = next(self.parameters())    
    return (weight.new_zeros(bsz,1, self.args.rnn_size),
            weight.new_zeros(bsz,1, self.args.rnn_size))


