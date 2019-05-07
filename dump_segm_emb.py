import sys
import torch
import numpy as np
from my_flags import *
from data_utils import *
from model_analizer import Analizer
from trainer import Trainer


if __name__ == '__main__':
  args = analizer_args()

  tbnames = open("tbnames_order_tab.txt",'r').read().strip("\n").split("\n")

  for tb in tbnames:
  	log_fn = "models-segm/"+tb+"/log.out"
  	ep_acc_line = open(log_fn,'r').read().strip("\n").split("\n")[-1]
  	ep,_ = ep_acc_line.split("\t")
  	input_model = "models-segm/%s/segm_%s.pth" % (tb,ep)

  	tb_args = args
  	tb_args.input_model = input_model
  	tb_args.train_file = "data/"+tb+"/train"

  	loader = DataLoaderAnalizer(tb_args)
  	state_dict = torch.load(tb_args.input_model)
  	emb_matrix = state_dict["emb.weight"]
  	outname = 
  	torch.save()



  
  
