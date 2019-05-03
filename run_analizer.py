import sys
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from torch.optim import Adam
from time import monotonic
from my_flags import *
from data_utils import *
from model_analizer import Analizer
from trainer import Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pdb


def train(args):
  print("Loading data...")
  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")
  print("Init batch objs")
  train_batch = BatchSegm(train,args.batch_size,args.gpu)
  dev_batch   = BatchSegm(dev,args.batch_size,args.gpu)
  n_vocab = loader.get_vocab_size()
  debug_print = int(100 / args.batch_size) + 1
  debug = True

  # init trainer
  model = Analizer(args,n_vocab)
  trainer = Trainer(model,n_vocab,args)
  
  for ep in range(args.epochs):
    train_loss = 0
    i = 0
    for sents,gold in train_batch.get_batch():
      train_loss += torch.sum(trainer.train_batch(sents, gold, debug=False))
      if i % debug_print == (debug_print - 1):
        print(".", end="", flush=True)
      i += 1
    #
    dev_loss = 0.0
    i = 0
    for sents,gold in dev_batch.get_batch():
      dev_loss += torch.sum(trainer.compute_loss(sents,gold,debug).data)
      if i % debug_print == (debug_print - 1):
          print(".", end="", flush=True)
      i += 1
    trainer.update_summary(train_loss,dev_loss,ep)

    finish_iter_time = monotonic()
    train_acc = evaluate_accuracy(model, train_data[:1000], batch_size, gpu)
    dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)





def main(args):
  print(args)
  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

  if args.mode == "train":
    train(args)
  else:
    test(args)
  




if __name__ == '__main__':
  args = analizer_args()
  sys.exit(main(args))
