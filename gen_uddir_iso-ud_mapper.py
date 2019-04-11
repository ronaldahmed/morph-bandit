import os
from utils import iso_mapper
import pdb

outfile_fn = "uddir_isoud_mapper.sh"

expand_iso = {v:k for k,v in iso_mapper.items()}


with open(outfile_fn,'w') as outfile:
  print("#!/bin/bash\n",file=outfile)
  print("declare -A uddir_isotb\n",file=outfile)

  for _,dnames,_ in os.walk("task2"):
    for ud_dir in dnames:
      ud_lang,tb = ud_dir.lower().split("-")
      lang = "_".join(ud_lang.split("_")[1:])

      if lang not in expand_iso:
        continue
      isotb = expand_iso[lang] + "_" + tb
      print('uddir_isotb["%s"]="%s"' % (ud_dir,isotb), file=outfile)

    #
  #