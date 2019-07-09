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
from model_analizer_bundle import AnalizerBundle

import sys,os
home = os.getenv("HOME")
sys.path.append(os.path.join(home,"MUSE"))


import pdb


class AnalizerSeq(AnalizerBundle):
  def __init__(self,args,nvocab):
    super(AnalizerSeq, self).__init__(args,nvocab)
    self.args = args
    self.nvocab = nvocab
    # self.cuda = to_cuda(args.gpu)
    self.rnn_hidden = None
    self.encoder = Encoder(args)
    self.decoder = AttDecoder(args,nvocab)
    self.init_hidden(args.batch_size)

  def init_body(self,args,nvocab):
    return None

  def forward(self, batch, dec_input, hidden):
    """ batch: Sx[bs x W] """
    encoder_output_hidden = self.encoder.forward(batch,hidden)
    dec_outputs = []
    k = 0
    for op_seq,enc_hid in encoder_output_hidden:
      output,hid,attn_w = self.decoder.forward(dec_input[k],enc_hid,op_seq)
      dec_outputs.append(output)
      k += 1
    #
    return dec_outputs


  def init_hidden(self, bsz):
    weight = next(self.encoder.parameters())
    if self.args.rnn_type == 'LSTM':
      self.rnn_hidden = [(self.cuda(weight.new_zeros(2,bsz,self.args.op_enc_size)),
                          self.cuda(weight.new_zeros(2,bsz,self.args.op_enc_size)) ),
                         (self.cuda(weight.new_zeros(2,bsz,self.args.w_enc_size)),
                          self.cuda(weight.new_zeros(2,bsz,self.args.w_enc_size)) )]
                         
    else:
      self.rnn_hidden = [self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size)),
                         self.cuda(weight.new_zeros(2,bsz,self.args.rnn_size)) ]
                        



class Encoder(Module):
  def __init__(self,args):
    super(Encoder, self).__init__()
    self.cuda = to_cuda(args.gpu)
    self.emb = self.load_embeddings(args)
    self.args = args
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


  def load_embeddings(self,args):
    if "models-segm" in self.args.embedding_pth:
      emb = self.cuda(nn.Embedding.from_pretrained(torch.load(args.embedding_pth).contiguous()) )
    else:
      emb = self.cuda(nn.Embedding.from_pretrained(torch.load(args.embedding_pth)["vectors"].contiguous()) )
    for param in emb.parameters():
      param.requires_grad = True
    return emb

  """
  Hierarchical (two levels) encoder
  1. biLSTM to encode actions
  2. [fw,bw] of each action -> w_repr
  3. biLSTM over w_repr to obtain h_w (introduces context into w repr)
  4. h_action <- [op_fw;op_bw;hw_fw;hw_bw]
  """
  def forward(self,batch,hidden):
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


    encoder_hid_st = ( torch.sum(hidden_w[0],0).view(1,batch_size,-1),
                       torch.sum(hidden_w[1],0).view(1,batch_size,-1)
                      )
    # hidw_fw = hidden_w[:,:,0,:].view(batch_size,-1,self.args.op_enc_size)
    # hidw_bw = hidden_w[:,:,1,:].view(batch_size,-1,self.args.op_enc_size)
    # encoder_hid_st = torch.cat([hidw_fw,hidw_bw],2)

    encoder_output = []

    # construct repr seq for each wop
    for i,op_fb in enumerate(op_fw_bw_seq):
      n_ops = op_fb.shape[1]
      h_tok_fw = h_w[:,i,0,:].view(batch_size,1,-1).repeat(1,n_ops,1)
      h_tok_bw = h_w[:,i,1,:].view(batch_size,1,-1).repeat(1,n_ops,1)
      # seq of actions w0,a1,a2: encoded seq
      ctx_action_seq = torch.cat([op_fb,h_tok_fw,h_tok_bw],2) # [fw,bw,w_fw,w_bw]

      encoder_output.append([ctx_action_seq,encoder_hid_st])
    return encoder_output    



class AttDecoder(Module):
  def __init__(self,args,nvocab):
    super(AttDecoder, self).__init__()
    ## FEATURE DECODER
    self.nvocab = nvocab
    self.args = args
    self.cuda = to_cuda(args.gpu)
    self.h_enc_size = 4 * args.op_enc_size
    self.dec_emb = self.cuda(nn.Embedding(nvocab, args.feat_dec_size))
    self.attn = self.cuda(nn.Linear(self.h_enc_size,args.feat_dec_size))
    self.attn_combine = self.cuda(nn.Linear(args.feat_dec_size + self.h_enc_size, args.feat_dec_size))
    self.dropout = nn.Dropout(args.dropout)
    self.decoder = self.cuda(
        getattr(nn, args.rnn_type)(args.op_enc_size,args.feat_dec_size,dropout=args.dropout,batch_first=True,bidirectional=False)
        )    
    self.out = self.cuda(nn.Linear(args.feat_dec_size, nvocab))


  def init_weights(self):
    emb_range = 1.0 / np.sqrt(self.nvocab)
    out_range = 1.0 / np.sqrt(2*self.args.feat_dec_size)
    self.dec_emb.weight.data.uniform_(-emb_range, emb_range)
    self.out.bias.data.zero_()
    self.out.weight.data.uniform_(-out_range, out_range)
    

  """
  Global attention - general (Luong et al 2016)
  """
  def forward(self,input,hidden,h_enc):
    batch_size = input.shape[0]
    embedded = self.dec_emb(input)
    embedded = self.dropout(embedded)
    
    h_t, hidden = self.decoder(embedded, hidden)

    he_Wa = self.attn(h_enc.view(-1,self.h_enc_size)).view(batch_size,-1,self.args.feat_dec_size)
    ht_Wa_he = torch.bmm(h_t,he_Wa.transpose(dim0=1,dim1=2))
    score = F.softmax(ht_Wa_he,-1) # bs x dec_len x enc_len
    c_attn = torch.bmm(score,h_enc)
    comb = torch.cat([h_t,c_attn],2).view(-1,self.args.feat_dec_size + self.h_enc_size)

    ht_hat = self.attn_combine(comb)
    ht_hat = F.relu(ht_hat)
    output = self.out(ht_hat).view(-1,self.nvocab)

    return output, hidden, score