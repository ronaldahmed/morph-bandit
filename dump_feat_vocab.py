import sys,os
import torch
import numpy as np
from data_utils import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from my_flags import analizer_args
from utils import saveObject

args = analizer_args()

tbnames = [x.strip('\n').split(" ")[1] for x in open("data/tbnames-thesis","r") if x.strip("\n")!=""]


for tb in tbnames:
    print(tb)

    log_fn = "models-segm/"+tb+"/log.out"
    ep_acc_line = open(log_fn,'r').read().strip("\n").split("\n")[-1]
    ep,_ = ep_acc_line.split("\t")
    input_model = "models-segm/%s/segm_%s.pth" % (tb,ep)

    tb_args = args
    tb_args.input_lem_model = input_model
    tb_args.train_file = "data/"+tb+"/train"
    tb_args.dev_file = "data/"+tb+"/dev"
    loader = DataLoaderAnalizer(tb_args)
    out_fn = "data/"+tb+"/feats.vocab"
    saveObject(loader.vocab_feats,out_fn)


# python dump_feat_vocab.py --mode dev \
# --train_file foo \
# --dev_file foo \
# --input_model foo
