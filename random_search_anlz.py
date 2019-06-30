import sys
import hyperopt
from hyperopt import hp, fmin, tpe, space_eval
import sys
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from torch.optim import Adam
from time import monotonic
from my_flags import *
from utils import STOP_LABEL
from data_utils import *
from model_analizer import Analizer
from model_lemmatizer import Lemmatizer
from trainer_analizer import TrainerAnalizer
from trainer_lemmatizer import TrainerLemmatizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pdb


def random_search(args):
  def train_search(curr_args):
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

    trainer_lem = TrainerLemmatizer(lemmatizer,n_vocab,args)
    trainer_analizer = TrainerAnalizer(analizer,n_feats,args)

    trainer_lem.freeze_model()
    trainer_lem.stop_id = loader.vocab_oplabel.get_label_id(STOP_LABEL)

    # init local vars
    best_dev_acc = -1

    for ep in range(curr_args.epochs):
      train_loss = 0
      i = 0
      start_time = monotonic()
      for bundle in train_batch.get_batch():
        loss = trainer_analizer.train_batch(bundle, debug=False)
        train_loss += loss
        # if i>3: break
        # i+=1
      #
      finish_iter_time = monotonic()
      train_loss /= train.get_num_instances()
      
      dev_metrics   = trainer_analizer.eval_metrics_batch(trainer_lem,dev_batch,loader,split="dev")
      dev_acc = dev_metrics.msd_f1

      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
    #
    return best_dev_acc
  ########
  
  def objective(opt_params):
    exp_args = args
    print(opt_params,"\n")

    exp_args.learning_rate = opt_params["lr"]
    exp_args.dropout = opt_params["dropout"]
    # exp_args.op_enc_size = int(opt_params["op_enc_size"])
    # exp_args.w_enc_size = int(opt_params["w_enc_size"])
    # exp_args.w_mlp_size = int(opt_params["w_mlp_size"])
    exp_args.clip = int(opt_params["clip"])
    exp_args.batch_size = int(opt_params["batch_size"])
    acc = train_search(exp_args)
    print("\n-->",acc,"\n",sep=" ")

    return -acc
  ##################################################

  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")

  train_batch = BatchAnalizer(train,args)
  dev_batch   = BatchAnalizer(dev,args)
  n_vocab = loader.get_vocab_size()
  n_feats = loader.get_feat_vocab_size()

  
  ###############

  space = {
    'lr': hp.loguniform('lr', -9, -1),
    'clip': hp.loguniform('clip', -2, 0),
    'dropout': hp.uniform('dropout', 0, 0.1),
    # 'w_enc_size': hp.quniform('w_enc_size', low=10, high=100,q=10),
    # 'w_mlp_size': hp.quniform('w_mlp_size', low=10, high=300,q=10),
    'batch_size': hp.quniform('batch_size', low=10, high=128,q=10)
  }
  
  best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
  
  print(best,"\n")
  # -> {'a': 1, 'c2': 0.01420615366247227}
  print(space_eval(space, best))

  

if __name__ == '__main__':
  args = analizer_args()
  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
  random_search(args)
