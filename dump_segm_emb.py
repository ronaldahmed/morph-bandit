import sys
import torch
import numpy as np


if __name__ == '__main__':
  # args = analizer_args()

  # tbnames = open("tbnames_order_tab.txt",'r').read().strip("\n").split("\n")
  tnames ="""
pt_gsd
no_nynorsk
fa_seraji
de_gsd
hy_armtdp""".strip("\n").split("\n")

  for tb in tbnames:
  	print(tb)

  	log_fn = "models-segm/"+tb+"/log.out"
  	ep_acc_line = open(log_fn,'r').read().strip("\n").split("\n")[-1]
  	ep,_ = ep_acc_line.split("\t")
  	input_model = "models-segm/%s/segm_%s.pth" % (tb,ep)

  	# tb_args = args
  	# tb_args.input_model = input_model
  	# tb_args.train_file = "data/"+tb+"/train"
  	# loader = DataLoaderAnalizer(tb_args)

  	state_dict = torch.load(input_model,map_location='cpu')
  	emb_matrix = state_dict["emb.weight"]
  	outname = "models-segm/%s/emb.pth" % (tb)
  	torch.save(emb_matrix,outname)




  
  
