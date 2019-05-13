import sys
import torch
import numpy as np
import hyperopt
from hyperopt import hp, fmin, tpe, space_eval
from time import monotonic
from my_flags import *
from data_utils import *
from model_lemmatizer import Lemmatizer
from trainer_lemmatizer import TrainerLemmatizer as Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pdb


def random_search(args):
  def train_search(curr_args):
    # init trainer
    model = Lemmatizer(curr_args,n_vocab)
    trainer = Trainer(model,n_vocab,curr_args)

    # init local vars
    best_dev_acc = -1

    for ep in range(curr_args.epochs):
      train_loss = 0
      i = 0
      start_time = monotonic()
      for sents,gold in train_batch.get_batch():
        loss = trainer.train_batch(sents, gold, debug=False)
        train_loss += loss
        # if i>3: break
        # i+=1
      #
      finish_iter_time = monotonic()
      train_loss /= train.get_num_instances()
      train_acc,train_dist = trainer.eval_metrics_batch(train_batch,loader,split="train",max_data=1000)
      try:
        dev_acc  ,dev_dist   = trainer.eval_metrics_batch(dev_batch, loader,split="dev")
      except:
        print("\n nn diverged!!\n")
        dev_acc,dev_dist = 0,-1
        break
      print(  "\nEpoch {:>4,} train | time: {:>9,.3f}m, loss: {:>12,.3f}, acc: {:>8,.3f}%, dist: {:>8,.3f}\n"
            "           dev   | time: {:>9,.3f}m, loss: {:>12,.3f}, acc: {:>8,.3f}%, dist: {:>8,.3f}\n"
            .format(ep,
                    (finish_iter_time - start_time) / 60,
                    train_loss,
                    train_acc,
                    train_dist,
                    (monotonic() - finish_iter_time) / 60,
                    0.0,
                    dev_acc,
                    dev_dist)
        )

      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
    #
    return best_dev_acc
  ########
  
  def objective(opt_params):
    exp_args = args
    print("\n")
    print(opt_params)

    exp_args.learning_rate = opt_params["lr"]
    exp_args.dropout = opt_params["dropout"]
    exp_args.clip = opt_params["clip"]
    exp_args.emb_size = int(opt_params["emb_size"])
    exp_args.mlp_size = int(opt_params["mlp_size"])
    exp_args.batch_size = int(opt_params["batch_size"])
    acc = train_search(exp_args)
    print("\n-->",acc)

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
    'lr': hp.loguniform('lr', -9, -3),
    'dropout': hp.uniform('dropout', 0, 0.2),
    'clip': hp.loguniform('clip', -4, 0),
    'emb_size': hp.quniform('emb_size', low=50, high=100,q=5),
    'mlp_size': hp.quniform('mlp_size', low=10, high=100,q=10),
    'batch_size': hp.quniform('batch_size', low=10, high=32,q=3)
  }
  
  best = fmin(objective, space, algo=tpe.suggest, max_evals=10)
  
  print(best)
  # -> {'a': 1, 'c2': 0.01420615366247227}
  print(space_eval(space, best))

  

if __name__ == '__main__':
  args = lemmatizer_args()
  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
  random_search(args)
