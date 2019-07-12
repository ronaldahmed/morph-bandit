import sys,os
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
  train_batch = BatchAnalizer(train,args)
  dev_batch   = BatchAnalizer(dev,args)
  n_vocab = loader.get_vocab_size()
  n_feats = loader.get_feat_vocab_size()

  print("vocab:",n_vocab," - feats:",n_feats)

  debug_print = int(100 / args.batch_size) + 1
  train_log_step_cnt = 0
  debug = True

  # init trainer
  lemmatizer = Lemmatizer(args,n_vocab)  
  analizer = Analizer(args,n_feats)
  
  # load lemmatizer
  if args.input_lem_model == "-":
    print("Please specify lemmatizer model to load!")
    return
  if args.gpu:
    state_dict = torch.load(args.input_lem_model)
  else:
    state_dict = torch.load(args.input_lem_model, map_location=lambda storage, loc: storage)
  lemmatizer.load_state_dict(state_dict)

  trainer_lem = TrainerLemmatizer(lemmatizer,loader,args)  
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
    for bundle in train_batch.get_batch():
      loss = trainer_analizer.train_batch(bundle, debug=False)
      train_loss += loss

      if i % debug_print == (debug_print - 1):
        trainer_analizer.update_summary(train_log_step_cnt,train_loss=loss)
        print(".", end="", flush=True)
      i += 1
      train_log_step_cnt += 1

      # if i>10: break
    #
    dev_loss = 0.0
    i = 0
    for bundle in dev_batch.get_batch(shuffle=False):
      dev_loss += trainer_analizer.eval_batch(bundle,debug=False)
      if i % debug_print == (debug_print - 1):
          print(".", end="", flush=True)
      i += 1

      # if i>5: break
    #
    dev_loss /= dev.get_num_instances()
    train_loss /= train.get_num_instances()

    finish_iter_time = monotonic()
    train_metrics = trainer_analizer.eval_metrics_batch(trainer_lem,train_batch,loader,split="train",max_data=1000)
    dev_metrics   = trainer_analizer.eval_metrics_batch(trainer_lem,dev_batch  ,loader,split="dev")
    dev_acc = dev_metrics.msd_f1
    
    trainer_analizer.update_summary(train_log_step_cnt,train_loss,dev_loss,
                                    train_metrics=train_metrics,dev_metrics=dev_metrics)

    print(  "\nEpoch {:>4,} train | time: {:>4,.3f}m, loss: {:>8,.3f}, acc: {:>6,.2f}%, dist: {:>6,.3f}, msd_acc: {:>6,.2f}, msd_f1: {:>6,.2f}\n"
            "           dev   | time: {:>4,.3f}m, loss: {:>8,.3f}, acc: {:>6,.2f}%, dist: {:>6,.3f}, msd_acc: {:>6,.2f}, msd_f1: {:>6,.2f}\n"
            .format(ep,
                    (finish_iter_time - start_time) / 60,
                    train_loss,
                    train_metrics.lem_acc,
                    train_metrics.lem_edist,
                    train_metrics.msd_acc,
                    train_metrics.msd_f1,
                    (monotonic() - finish_iter_time) / 60,
                    dev_loss,
                    dev_metrics.lem_acc,
                    dev_metrics.lem_edist,
                    dev_metrics.msd_acc,
                    dev_metrics.msd_f1)
        )
    if dev_loss < best_dev_loss:
      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_ep = ep
        print("New best acc!")
      print("New best dev!")
      best_dev_loss = dev_loss
      best_dev_loss_index = 0
      trainer_analizer.save_model(ep)
        
    else:
      best_dev_loss_index += 1
      if best_dev_loss_index == args.patience:
        print("Reached", args.patience, "iterations without improving dev loss. Breaking")
        break
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_ep = ep
      print("New best acc!")
      trainer_analizer.save_model(ep)

    if trainer_analizer.scheduler != None:
      trainer_analizer.scheduler.step(dev_loss)
    #
  #
  print(best_ep,best_dev_acc,sep="\t")


#################################################################################################################


def train_simple(args):
  """ Used for multi-seeding analysis
  - does not save parameters
  - doesn't calc metrics/loss on the training set
  """
  tbname = os.path.basename(os.path.dirname(args.train_file))
  prefix_output_name = "models-ens/%s/anlz-%d" % (tbname,args.seed)

  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data("dev")

  train_batch = BatchAnalizer(train,args)
  dev_batch   = BatchAnalizer(dev,args)
  n_vocab = loader.get_vocab_size()
  n_feats = loader.get_feat_vocab_size()

  debug_print = int(100 / args.batch_size) + 1
  train_log_step_cnt = 0
  debug = True

  # init trainer
  lemmatizer = Lemmatizer(args,n_vocab)
  analizer = Analizer(args,n_feats)

  # load lemmatizer
  if args.input_lem_model == "-":
    print("Please specify lemmatizer model to load!")
    return
  if args.gpu:
    state_dict = torch.load(args.input_lem_model)
  else:
    state_dict = torch.load(args.input_lem_model, map_location=lambda storage, loc: storage)
  lemmatizer.load_state_dict(state_dict)

  trainer_lem = TrainerLemmatizer(lemmatizer,loader,args)
  trainer_analizer = TrainerAnalizer(analizer,n_feats,args)
  trainer_lem.freeze_model()

  # <-----------------

  # init local vars
  best_dev_loss = 100000000
  best_dev_loss_index = -1
  best_dev_acc = -1
  best_ep = -1
  best_metrics = None

  for ep in range(args.epochs):
    train_loss = 0
    i = 0
    for sents,gold in train_batch.get_batch():
      loss = trainer_analizer.train_batch(sents, gold, debug=False)

    #
    dev_loss = 0.0
    for sents,gold in dev_batch.get_batch(shuffle=False):
      dev_loss += trainer_analizer.eval_batch(sents,gold,debug=False)
    dev_loss /= dev.get_num_instances()

    dev_metrics   = trainer_analizer.eval_metrics_batch(trainer_lem,dev_batch,loader,split="dev",
                        dump_ops=False,
                        output_name=prefix_output_name)
    dev_acc = dev_metrics.msd_f1
    
    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_ep = ep
      best_metrics = dev_metrics

    if trainer_analizer.scheduler != None:
      trainer_analizer.scheduler.step(dev_loss)
    print("Ep: %d | loss:%.4f | lem_acc:%.4f | lem_edist:%.4f | msd_acc:%.4f | msd_f1:%.4f" % 
          (ep,dev_loss,
            dev_metrics.lem_acc,
            dev_metrics.lem_edist,
            dev_metrics.msd_acc,
            dev_metrics.msd_f1) )
    #
  #
  print(best_ep,
        best_metrics.lem_acc,
        best_metrics.lem_edist,
        best_metrics.msd_acc,
        best_metrics.msd_f1,
        sep="\t")





#################################################################################################################

def test(args):
  print("Loading data...")
  to_eval_split = "dev" if args.mode=="dev" else "test"
  loader = DataLoaderAnalizer(args)
  train = loader.load_data("train")
  dev   = loader.load_data(to_eval_split)

  print("Init batch objs")
  train_batch = BatchAnalizer(train,args)
  dev_batch   = BatchAnalizer(dev,args)
  n_vocab = loader.get_vocab_size()
  n_feats = loader.get_feat_vocab_size()

  debug_print = int(100 / args.batch_size) + 1
  train_log_step_cnt = 0
  debug = True

  # init trainer
  lemmatizer = Lemmatizer(args,n_vocab)
  analizer = Analizer(args,n_feats)

  # load lemmatizer
  if args.input_lem_model == "-":
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
  start_time = monotonic()
  dev_metrics   = trainer_analizer.eval_metrics_batch(
                                      trainer_lem,
                                      dev_batch,
                                      loader,
                                      split=to_eval_split,
                                      covered=(args.mode=="covered-test"),
                                      dump_ops=args.dump_ops)
  print("time: ",(monotonic() - start_time)/60.0)
  print("%s | lem_acc: %.4f, dist: %.4f, msd_acc: %.4f, msd_f1: %.4f" % 
                (to_eval_split,
                  dev_metrics.lem_acc,
                  dev_metrics.lem_edist,
                  dev_metrics.msd_acc,
                  dev_metrics.msd_f1 ))
  return



def main(args):
  print(args)
  if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

  if args.mode == "train":
    train(args)
  elif args.mode == "train_simple":
    train_simple(args)
  else:
    test(args)
  


if __name__ == '__main__':
  args = analizer_args()
  sys.exit(main(args))
