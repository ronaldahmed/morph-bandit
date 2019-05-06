import sys
import torch
import numpy as np
import hyperopt
from hyperopt import hp, fmin, tpe, space_eval
from my_flags import *
from data_utils import *
from model_analizer import Analizer
from trainer import Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pdb


def random_search(args):
  def train_search(curr_args):
    # init trainer
    model = Analizer(curr_args,n_vocab)
    trainer = Trainer(model,n_vocab,curr_args)

    # init local vars
    best_dev_acc = -1

    for ep in range(curr_args.epochs):
      train_loss = 0
      i = 0
      for sents,gold in train_batch.get_batch():
        loss = torch.sum(trainer.train_batch(sents, gold, debug=False))
        train_loss += loss
        # if i>3: break
        # i+=1
      #
      
      train_acc,train_dist = trainer.eval_metrics_batch(train_batch,loader,split="train",max_data=1000)
      dev_acc  ,dev_dist   = trainer.eval_metrics_batch(dev_batch, loader,split="dev")
      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
    #
    return best_dev_acc
  ########
  
  def objective(opt_params):
    exp_args = args
    print(opt_params)

    exp_args.learning_rate = opt_params["lr"]
    exp_args.dropout = opt_params["dropout"]
    exp_args.emb_size = int(opt_params["emb_size"])
    exp_args.mlp_size = int(opt_params["mlp_size"])
    exp_args.batch_size = int(opt_params["batch_size"])
    acc = train_search(exp_args)
    print("-->",acc,sep=" ")

    return -acc
  ##################################################

  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")

  train_batch = BatchSegm(train,args.batch_size,args.gpu)
  dev_batch   = BatchSegm(dev,args.batch_size,args.gpu)
  n_vocab = loader.get_vocab_size()

  
  ###############

  space = {
    'lr': hp.loguniform('lr', -9, -2),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'emb_size': hp.quniform('emb_size', low=50, high=300,q=20),
    'mlp_size': hp.quniform('mlp_size', low=100, high=300,q=10),
    'batch_size': hp.quniform('batch_size', low=10, high=128,q=10)
  }
  
  best = fmin(objective, space, algo=tpe.suggest, max_evals=30)
  
  print(best)
  # -> {'a': 1, 'c2': 0.01420615366247227}
  print(space_eval(space, best))

  

if __name__ == '__main__':
  args = analizer_args()
  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
  random_search(args)
