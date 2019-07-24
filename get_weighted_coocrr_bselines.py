import sys
import torch
import numpy as np
from time import monotonic
from my_flags import *
from data_utils import *
from model_analizer import Analizer
from model_lemmatizer import Lemmatizer
from trainer_analizer import TrainerAnalizer
from trainer_lemmatizer import TrainerLemmatizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict, Counter
from utils import STOP_LABEL, SPACE_LABEL, apply_operations

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family' : 'serif',
        'serif': 'Times',
        'size'   : 11}
matplotlib.rc('font', **font)


import pdb

def get_top_freq(loader,datawrap,max_ops=50,max_fts=50):
  op_counter = Counter()
  feat_counter = Counter()
  for sent,feats in zip(datawrap.ops,datawrap.feats):
    for op_seq,feat in zip(sent,feats):
      # print([loader.vocab_oplabel.get_label_name(_id) for _id in op_seq])
      # pdb.set_trace()
      op_counter.update(op_seq[1:-1]) # don't consider start
      feat_label = loader.get_feat_label(feat)
      indv_labels = feat_label.split(";")
      feat_counter.update(indv_labels)
  #

  top_ops = set([x for x,y in op_counter.most_common(max_ops)])
  top_fts = set([x for x,y in feat_counter.most_common(max_fts)])
  return top_ops,top_fts


def get_gold_ditribution(loader,dev,op_mapper,ft_mapper):
  n_ops = len(op_mapper)
  p_op_ft = np.zeros([n_ops,len(ft_mapper)],dtype=float)

  for sent,feats_sent in zip(dev.ops,dev.feats):
    for op_seq,feat in zip(sent,feats_sent):
      feat_label = loader.get_feat_label(feat)
      indv_labels = feat_label.split(";")
      op_ids = [op_mapper[x] for x in op_seq if x in op_mapper]
      ft_ids = [ft_mapper[x] for x in indv_labels if x in ft_mapper]
      if len(op_ids) > 0 and len(ft_ids) > 0:
        for i_op in op_ids:
          p_op_ft[i_op,ft_ids] += 1.0     
  #
  for i in range(n_ops):
    p_op_ft[i,:] /= p_op_ft[i,:].sum()
  
  return p_op_ft


def get_pred_distribution(args,train_lem,train_anlz,loader,dev,op_mapper,ft_mapper):
  n_ops = len(op_mapper)
  p_op_ft = np.zeros([n_ops,len(ft_mapper)],dtype=float)

  batch   = BatchAnalizer(dev,args)
  stop_id = loader.vocab_oplabel.get_label_id(STOP_LABEL)

  # similar to evaluation / decoding on eval_metrics_batch
  for op_seqs,feats,forms,lemmas in batch.get_eval_batch():
    filtered_op_batch = []             # bs x [ S x W ]
    filt_score_batch  = []             # bs x [ S x W ]

    # 1. predict operation sequence
    predicted,scores = train_lem.predict_batch(op_seqs,start=True,score=True) # Sx[ bs x W ]
    predicted = batch.restore_batch(predicted)     # bs x [ SxW ]
    scores = batch.restore_batch(scores)     # bs x [ SxW ]

    #    get op labels & apply oracle
    for i,sent in enumerate(predicted):
      sent = predicted[i]
      scores_sent = scores[i]
      filt_op_sent = []
      filt_sc_sent = []
      len_sent = len(forms[i]) # forms and lemmas are not sent-padded
      for j in range(len_sent):
        w_op_seq = sent[j]
        w_sc_seq = scores_sent[j]
        form_str = forms[i][j].replace(SPACE_LABEL," ")
        if sum(w_op_seq)==0:
          pred_lemmas.append(form_str.lower())
          continue
        if stop_id in w_op_seq:
          _id = np.where(np.array(w_op_seq)==stop_id)[0][0]
          w_op_seq = w_op_seq[:_id+1]
        optokens = [loader.vocab_oplabel.get_label_name(x) \
                        for x in w_op_seq if x!=PAD_ID]
        pred_lem,op_len = apply_operations(form_str,optokens)
        filt_op_sent.append( w_op_seq[:op_len+1].tolist() ) # discarded the stop_id
        filt_sc_sent.append( w_sc_seq[:op_len].tolist() )
      #
      filtered_op_batch.append(filt_op_sent)
      filt_score_batch.append(filt_sc_sent)
    #
    #  rebatch op seqs
    padded = batch.pad_data_per_batch(filtered_op_batch,[np.arange(len(filtered_op_batch))])
    reinv_op_batch = batch.invert_axes(padded,np.arange(len(filtered_op_batch))) # Sx[ bs x W ]

    # 2. predict labels
    pred_labels,pred_sc_batch = train_anlz.predict_batch(reinv_op_batch,score=True) # [bs x S]
    bs = pred_labels.shape[0]
    pred_feats = []
    pred_scores = []
    for i in range(bs):
      len_sent = len(forms[i])
      pred_feats.append(pred_labels[i,:len_sent])
      pred_scores.append(pred_sc_batch[i,:len_sent].tolist())
    #
    ##
    for sent_op,sent_sc,sent_ft,sent_ft_sc in \
                            zip(filtered_op_batch,filt_score_batch,
                                pred_feats,pred_scores):

      for pred_op_seq,sc_seq,pred_ft,pred_ft_sc in zip(sent_op,sent_sc,sent_ft,sent_ft_sc):
        pred_op_seq = pred_op_seq[1:]
        feat_label = loader.get_feat_label(pred_ft)
        indv_labels = feat_label.split(";")
        op_ids = [op_mapper[x] for x in pred_op_seq if x in op_mapper]
        op_scs = [sc for x,sc in zip(pred_op_seq,sc_seq) if x in op_mapper]
        ft_ids = [ft_mapper[x] for x in indv_labels if x in ft_mapper]

        if len(op_ids) > 0 and len(ft_ids) > 0:
          for op_id,op_sc in zip(op_ids,op_scs):
            p_op_ft[op_id,ft_ids] += op_sc * pred_ft_sc
        #
    #
    for i in range(n_ops):
      p_op_ft[i,:] = p_op_ft[i,:] / p_op_ft[i,:].sum() if p_op_ft[i,:].sum() > 0 else 0.0
    
    return p_op_ft


def plot_heatmaps(p_gold,p_pred,loader,op_mapper,ft_mapper):

  # sns.set()
  oplab_index = ['']*len(op_mapper)
  ftlab_index = ['']*len(ft_mapper)
  
  for x,index in op_mapper.items():
    oplab_index[index] = loader.vocab_oplabel.get_label_name(x)
  for x,index in ft_mapper.items():
    ftlab_index[index] = x
  
  gold_df = pd.DataFrame(data=p_gold,columns=ft_mapper,index=oplab_index)
  pred_df = pd.DataFrame(data=p_pred,columns=ft_mapper,index=oplab_index)

  
  # plt.figure()
  # sns.heatmap(gold_df,xticklabels=True, yticklabels=True,
  #             cmap="Spectral",vmax=1.0,vmin=0.0,
  #             square=True)
  # plt.title("Operations vs Gold features")

  # plt.figure()
  # sns.heatmap(pred_df,xticklabels=True, yticklabels=True,
  #             cmap="Spectral",vmax=1.0,vmin=0.0,square=True)
  # plt.title("Operations vs Predicted features")

  grid_kws = {"width_ratios": [21,20,1], "wspace": .1, "hspace": .1}
  f, (a0,a1,cax) = plt.subplots(nrows=1,ncols=3,gridspec_kw=grid_kws)

  # cbar_ax = fig.add_axes([.905, .3, .05, .3])
  sns.heatmap(gold_df,xticklabels=True, yticklabels=True,
              cmap="Spectral",vmax=1.0,vmin=0.0,
              cbar=False,
              ax=a0,
              )
  a0.title.set_text("Gold features")

  sns.heatmap(pred_df,xticklabels=True, yticklabels=False,
              cmap="Spectral",vmax=1.0,vmin=0.0,
              ax=a1,cbar_ax=cax,
              )
  a1.title.set_text("Predicted features")
  
  plt.tight_layout()
  plt.show()

  print("-->")



def main(args):
  print(args)
  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

  ## similar to decoding

  print("Loading data...")
  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")

  print("Init batch objs")
  train_batch = BatchAnalizer(train,args)
  dev_batch   = BatchAnalizer(dev,args)
  n_vocab = loader.get_vocab_size()
  n_feats = loader.get_feat_vocab_size()


  # init trainer
  lemmatizer = Lemmatizer(args,n_vocab)
  analizer = Analizer(args,n_feats)

  # load lemmatizer
  if args.input_lem_model is None:
    print("Please specify lemmatizer model to load!")
    return
  if args.gpu:
    state_dict = torch.load(args.input_lem_model)
  else:
    state_dict = torch.load(args.input_lem_model, map_location=lambda storage, loc: storage)
  lemmatizer.load_state_dict(state_dict)

  trainer_lem = TrainerLemmatizer(lemmatizer,loader,args)
  trainer_lem.freeze_model()

  # load analizer
  if args.input_model is None:
    print("Please specify model to load!")
    return
  if args.gpu:
    state_dict = torch.load(args.input_model)
  else:
    state_dict = torch.load(args.input_model, map_location=lambda storage, loc: storage)

  analizer.load_state_dict(state_dict)
  trainer_analizer = TrainerAnalizer(analizer,n_feats,args)
  ##########

  # 1. get filtered list of ops and individual features
  op_filt,feat_filt = get_top_freq(loader,train,max_ops=50,max_fts=50)
  op_mapper = {x:i for i,x in enumerate(op_filt)}
  ft_mapper = {x:i for i,x in enumerate(feat_filt)}


  # 2. get gold counts dev set
  p_gold = get_gold_ditribution(loader,dev,op_mapper,ft_mapper)

  # 3. get nn - weighted prob distr on dev set
  p_pred = get_pred_distribution(args,trainer_lem,trainer_analizer, \
                                 loader,dev,op_mapper,ft_mapper)

  plot_heatmaps(p_gold,p_pred,loader,op_mapper,ft_mapper)

  # pdb.set_trace()
  

if __name__ == '__main__':
  args = analizer_args()
  sys.exit(main(args))
