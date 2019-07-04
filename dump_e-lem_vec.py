import sys
import torch
import numpy as np
from my_flags import *
from data_utils import *

import pdb

if __name__ == '__main__':
  args = analizer_args()

  tbnames = [x.strip('\n').split(" ")[1] for x in open("data/tbnames-thesis","r") if x.strip("\n")!=""]
  # tbnames = ["es_ancora"]
  
  for tb in tbnames:
    print(tb)

    log_fn = "models-segm/"+tb+"/log.out"
    ep_acc_line = open(log_fn,'r').read().strip("\n").split("\n")[-1]
    ep,_ = ep_acc_line.split("\t")
    input_model = "models-segm/%s/segm_%s.pth" % (tb,ep)

    tb_args = args
    tb_args.input_model = input_model
    tb_args.train_file = "data/"+tb+"/train"
    tb_args.dev_file = "data/"+tb+"/dev"
    loader = DataLoaderAnalizer(tb_args)

    with open("l1-mono-emb/%s.vec" % tb,"w") as outfile:
      state_dict = torch.load(input_model,map_location='cpu')
      emb_matrix = state_dict["emb.weight"] # [vocab x emb_size]
      emb_matrix = emb_matrix.cpu().numpy()
      vocab_size,esize = emb_matrix.shape
      for i in range(vocab_size):
        tok = loader.vocab_oplabel.get_label_name(i)
        emb = " ".join(["%.5f"%x for x in emb_matrix[i,:]])
        print(tok,emb,sep=" ",file=outfile)


