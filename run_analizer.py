import sys
import torch
import numpy as np
from torch.nn import Module, Parameter, NLLLoss, LSTM
from torch.optim import Adam
from time import monotonic
from my_flags import *
from data_utils import *
from model_analizer import Analizer
from model_lemmatizer import Lemmatizer
from trainer_analizer import TrainerAnalizer
from trainer_lemmatizer import TrainerLemmatizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pdb


def train(args):
  print("Loading data...")
  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")

  print("Init batch objs")
  train_batch = BatchAnalizer(train,args.batch_size,args.gpu)
  dev_batch   = BatchAnalizer(dev,args.batch_size,args.gpu)
  n_vocab = loader.get_vocab_size()
  n_feats = loader.get_feat_vocab_size()

  debug_print = int(100 / args.batch_size) + 1
  train_log_step_cnt = 0
  debug = True

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
  
  # <-----------------

  # init local vars
  best_dev_loss = 100000000
  best_dev_loss_index = -1
  best_dev_acc = -1
  best_ep = -1

  for ep in range(args.epochs):
    start_time = monotonic()
    train_loss = 0
    i = 0
    for sents,gold in train_batch.get_batch():
      loss = trainer.train_batch(sents, gold, debug=False)
      train_loss += loss

      if i % debug_print == (debug_print - 1):
        trainer_analizer.update_summary(train_log_step_cnt,train_loss=loss)
        print(".", end="", flush=True)
      i += 1
      train_log_step_cnt += 1

      # if i>10: break
    
    dev_loss = 0.0
    i = 0
    for sents,gold in dev_batch.get_batch(shuffle=False):
      dev_loss += trainer.eval_batch(sents,gold,debug=False)
      if i % debug_print == (debug_print - 1):
          print(".", end="", flush=True)
      i += 1

      # if i>5: break
    #
    dev_loss /= dev.get_num_instances()
    train_loss /= train.get_num_instances()

    finish_iter_time = monotonic()
    train_metrics = trainer_analizer.eval_metrics_batch(trainer_lemmatizer,train_batch,loader,split="train",max_data=1000)
    dev_metrics   = trainer_analizer.eval_metrics_batch(trainer_lemmatizer,dev_batch  ,loader,split="dev")
    
    trainer_analizer.update_summary(train_log_step_cnt,train_loss,dev_loss,
                                    train_metrics,dev_metrics)

    print(  "\nEpoch {:>4,} train | time: {:>9,.3f}m, loss: {:>12,.3f}, acc: {:>8,.3f}%, dist: {:>8,.3f}\n"
            "           dev   | time: {:>9,.3f}m, loss: {:>12,.3f}, acc: {:>8,.3f}%, dist: {:>8,.3f}\n"
            .format(ep,
                    (finish_iter_time - start_time) / 60,
                    train_loss,
                    train_acc,
                    train_dist,
                    (monotonic() - finish_iter_time) / 60,
                    dev_loss,
                    dev_acc,
                    dev_dist)
        )

    if dev_loss < best_dev_loss:
      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_ep = ep
        print("New best acc!")
      print("New best dev!")
      best_dev_loss = dev_loss
      best_dev_loss_index = 0
      trainer.save_model(ep)
        
    else:
      best_dev_loss_index += 1
      if best_dev_loss_index == args.patience:
        print("Reached", patience, "iterations without improving dev loss. Breaking")
        break
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_ep = ep
      print("New best acc!")
      trainer.save_model(ep)

    if trainer.scheduler != None:
      trainer.scheduler.step(dev_loss)
    #
  #
  print(best_ep,best_dev_acc,sep="\t")


def test(args):
  print("Loading data...")
  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  to_eval_split = "dev" if args.mode=="dev" else "test"
  dev   = loader.load_data(to_eval_split)

  print("Init batch objs")
  train_batch = BatchSegm(train,args.batch_size,args.gpu)
  dev_batch   = BatchSegm(dev,args.batch_size,args.gpu)
  n_vocab = loader.get_vocab_size()
  
  # init model
  model = Lemmatizer(args,n_vocab)
  # load model
  if args.input_model is None:
    print("Please specify model to load!")
    return
  if args.gpu:
    state_dict = torch.load(args.input_model)
  else:
    state_dict = torch.load(args.input_model, map_location=lambda storage, loc: storage)

  model.load_state_dict(state_dict)
  # if args.gpu:
  #   model.cuda(model)
  # init trainer
  trainer = Trainer(model,n_vocab,args)
  dev_acc  ,dev_dist   = trainer.eval_metrics_batch(
                                    dev_batch,
                                    loader,
                                    split=to_eval_split,
                                    covered=(args.mode=="covered-test"))
  print("%s | acc: %.4f, dist: %.4f" % (to_eval_split,dev_acc,dev_dist))
  return



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
