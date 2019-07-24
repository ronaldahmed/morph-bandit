import os,sys
import argparse
from collections import Counter, defaultdict
from my_flags import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import START, UNK_TOKEN, map_ud_folders
from data_utils import *
import pdb

# no sentence distinction
# just list of (form,lemma) tuples
# file is CONLLU format
def get_form_lemmas(filename):
  data = []
  for line in open(filename,'r'):
    line = line.strip('\n')
    if line=='' or line.startswith("#"): continue
    comps = line.split('\t')
    if len(comps)<2: continue
    w = comps[1].lower()
    l = comps[2].lower()
    data.append([w,l])
  return data

def get_form_lemma_mapper(tup_list):
  form_lemms_map = {}
  for w,lm in tup_list:
    if w not in form_lemms_map:
      form_lemms_map[w] = set()
    form_lemms_map[w].add(lm)
  return form_lemms_map

def get_custom_acc(gold_tup,pred_tups,licited):
  acc = 0.0
  total = 0
  for gold,pred in zip(gold_tups,pred_tups):
    try:
      gw,gl = gold
      pw,pl = pred
    except:
      pdb.set_trace()

    if gw != pw:
      print("diff words!!",gw,pw)
      pdb.set_trace()
    if gw not in licited: continue
    acc += int(gl==pl)
    total += 1
  return (100.0*acc) / total


def error_anlz_args():
  p = ArgumentParser(add_help=False)
  p.add_argument("--src_exp", "-s", help="Reference output to compare against", type=str, default=None)
  p.add_argument("--tgt_exp", "-t", help="our model's output", type=str, default=None)
  return p.parse_args()


def reformat_action(word):
  return "%s.%s-%s" % (START,START,word.lower())


if __name__ == '__main__':
  args = error_anlz_args()
  
  tb2ud = {y:x for x,y in map_ud_folders().items()}

  tbnames = [
    'es_ancora',
    'en_ewt',
    'cs_pdt',
    'tr_imst',
    'ar_padt',
    'de_gsd',
  ]

  exp_temp = "../thesis-files/models_pred/%s-um-%s.conllu.%s.pred"
  gold_temp = "2019/task2/%s/%s-um-%s.conllu"


  print("treebank,model,ambiguous,unseen,seen-unamb")

  for tb in tbnames:
    uddir = tb2ud[tb]
    # print(":: ",uddir,tb)
    # build filenames
    
    #
    train_fn = gold_temp % (uddir,tb,"train")
    gold_ref_fn = gold_temp % (uddir,tb,"dev")
    src_fn = exp_temp % (tb,"dev",args.src_exp)
    tgt_fn = exp_temp % (tb,"dev",args.tgt_exp)



    train_tups = get_form_lemmas(train_fn)
    gold_tups = get_form_lemmas(gold_ref_fn)
    src_tups = get_form_lemmas(src_fn)
    tgt_tups = get_form_lemmas(tgt_fn)

    train_mapper = get_form_lemma_mapper(train_tups)
    gold_mapper = get_form_lemma_mapper(gold_tups)
    src_mapper = get_form_lemma_mapper(src_tups)
    tgt_mapper = get_form_lemma_mapper(tgt_tups)

    joint_keys = set(list(train_mapper.keys()) + list(gold_mapper.keys()))
    joint_map = {x:train_mapper.get(x,set()) | gold_mapper.get(x,set()) for x in joint_keys}

    # ambiguous : train+dev
    amb_forms = set([x for x,y in joint_map.items() if len(y)>1])
    src_amb_acc = get_custom_acc(gold_tups,src_tups,amb_forms)
    tgt_amb_acc = get_custom_acc(gold_tups,tgt_tups,amb_forms)

    # unseen: w.r.t. train
    unseen_forms = set([x for x,y in gold_mapper.items() if x not in train_mapper])

    src_uns_acc = get_custom_acc(gold_tups,src_tups,unseen_forms)
    tgt_uns_acc = get_custom_acc(gold_tups,tgt_tups,unseen_forms)

    # seen unambiguous
    seen_unamb_forms = set([x for x,y in joint_map.items() if len(y)==1 and x not in unseen_forms])
    src_su_acc = get_custom_acc(gold_tups,src_tups,seen_unamb_forms)
    tgt_su_acc = get_custom_acc(gold_tups,tgt_tups,seen_unamb_forms)

    print(tb,args.src_exp,src_amb_acc,src_uns_acc,src_su_acc,sep=",")
    print(tb,args.tgt_exp,tgt_amb_acc,tgt_uns_acc,tgt_su_acc,sep=",")

    # print(src_amb_acc,tgt_amb_acc)
    # print(src_uns_acc,tgt_uns_acc)
    # print(src_su_acc,tgt_su_acc)
    # pdb.set_trace()

