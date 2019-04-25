import os
import pickle
import glob as gb

import pdb

# label names
MAX_INT = 1000000
INS = "ins"
DEL = "del"
SUBS = "subs"
TRSP = "trsp"
SKIP = "skip"
STOP = "STOP"
START = "START"

iso_mapper = {
  'en': 'english',
  'es': 'spanish',
  'da': 'danish',
  'cs': 'czech',
  'nl': 'dutch',
  'tr': 'turkish',
  'ar': 'arabic',
  'it': 'italian',
  'ja': 'japanese',
  'de': 'german',
  'sk': 'shk',
  'mt': 'maltese'
}



def saveObject(obj, name='model'):
  with open(name + '.pickle', 'wb') as fd:
    pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
  # Load tagger
  with open(obj_name, 'rb') as fd:
    obj = pickle.load(fd)
  return obj


def map_ud_folders():
  mapper = {}
  basedir = "2019/task2"
  for root,dirnames,_ in os.walk(basedir):
    for uddir in dirnames:
      _file_list = gb.glob(os.path.join(basedir,uddir,"*-um-*.conllu"))
      if len(_file_list)==0:
        continue
      tb_name = os.path.basename(_file_list[0]).split("-")[0]
      mapper[uddir] = tb_name
    break
  #

  return mapper