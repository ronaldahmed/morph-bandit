import os
import pickle
import glob as gb
import re
import pdb
import torch

# label names
MAX_INT = 1000000
INS = "ins"
DEL = "del"
SUBS = "subs"
TRSP = "trsp"
SKIP = "skip"
STOP = "STOP"
START = "START"

PREF_POS="_A"
SUFF_POS="A_"

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

oplabel_pat = re.compile("(?P<name>[a-zA-Z]+)[.](?P<pos>[_]?[0-9A-Z]+[_]?)-(?P<seg>.+)",re.UNICODE)

# Sopa sharing params
SHARED_SL_PARAM_PER_STATE_PER_PATTERN = 1
SHARED_SL_SINGLE_PARAM = 2


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


def to_cuda(gpu):
  return (lambda v: v.cuda()) if gpu else identity

def identity(x):
    return x

def fixed_var(tensor):
  return tensor.detach()


def apply_operations(init_form,operations,debug=False):
  """ Apply sequence of operations on initial form """
  curr_tok = init_form.lower()
  if debug:
    print(init_form,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

  for op_token in operations:
    match = oplabel_pat.match(op_token)
    if match==None:
      print("Operation token with bad format!!")
      pdb.set_trace()
    name = match.group("name")
    pos = match.group("pos")
    segment = match.group("seg")
    if name==START:
      continue
    if name==STOP:
      break

    if   pos == PREF_POS:
      if   name==INS:
        curr_tok = segment + curr_tok
      elif name==DEL:
        assert curr_tok.startswith(segment)
        curr_tok = curr_tok[len(segment):]
      elif name==SUBS:
        curr_tok = segment + curr_tok[len(segment):]
      elif name==TRSP:
        assert curr_tok.startswith(segment)
        assert len(segment)==2
        curr_tok = segment[::-1] + curr_tok[len(segment):]

    elif pos == SUFF_POS:
      if   name==INS:
        curr_tok += segment
      elif name==DEL:
        assert curr_tok.endswith(segment)
        curr_tok = curr_tok[:-len(segment)]
      elif name==SUBS:
        curr_tok = curr_tok[:-len(segment)] + segment
      elif name==TRSP:
        assert curr_tok.endswith(segment)
        assert len(segment)==2
        curr_tok = curr_tok[:-len(segment)] + segment[::-1]

    else:
      try:
        pos = int(pos[1:-1]) - 1
      except:
        print("bad pos:",segment)
        pdb.set_trace()

      if   name==INS:
        curr_tok = curr_tok[:pos] + segment + curr_tok[pos:]
      elif name==DEL:
        curr_tok = curr_tok[:pos] + curr_tok[pos+len(segment):]
      elif name==SUBS:
        curr_tok = curr_tok[:pos] + segment + curr_tok[pos+len(segment):]
      elif name==TRSP:
        assert curr_tok[pos:pos+len(segment)] == segment
        assert len(segment)==2
        curr_tok = curr_tok[:pos] + segment[::-1] + curr_tok[pos+len(segment):]
    #
    if debug:
      print("\t",op_token,"|",curr_tok)
  #
  
  return curr_tok
