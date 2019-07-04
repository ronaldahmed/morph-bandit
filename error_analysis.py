import os,sys
import argparse
from collections import Counter, defaultdict
from my_flags import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import START, UNK_TOKEN
from data_utils import *
import pdb

# no sentence distinction
# just list of (form,lemma) tuples
# file is CONLLU format
def get_form_lemmas(filename):
  data = []
  for line in open(filename,'r'):
    line = line.strip('\n')
    if line=='': continue
    comps = line.split('\t')
    if len(comps)<2: continue
    data.append(map(lambda x: x.lower(), comps[1:3]))
  return data

def get_form_lemma_mapper(tup_list):
  form_lemms_map = defaultdict(set)
  for w,lm in tup_list:
    form_lemms_map[w].add(lm)
  return form_lemms_map

def get_custom_acc(gold_tup,pred_tups,licited):
  acc = 0.0
  total = 0
  for gold,pred in zip(gold_tups,pred_tups):
    gw,gl = gold
    pw,pl = pred
    if gw != pw:
      print("diff words!!",gw,pw)
      pdb.set_trace()
    if gw not in licited: continue
    acc += int(gl==pl)
    total += 1
  return (1.0*acc) / total


def error_ref_anlz_args():
  p = ArgumentParser(add_help=False)
  p.add_argument("--src_ref", "-s", help="Reference output to compare against", type=str, default=None)
  p.add_argument("--tgt_ref", "-t", help="our model's output", type=str, default=None)
  return p

def error_anlz_args():
  parser = ArgumentParser(description=__doc__,
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          parents=[soft_pattern_arg_parser(), morph_analizer_arg_parser(), \
                                   lemmatizer_arg_parser(),  general_arg_parser(), error_ref_anlz_args()])
  return parser.parse_args()

def reformat_action(word):
  return "%s.%s-%s" % (START,START,word.lower())


if __name__ == '__main__':
  args = error_anlz_args()
  # ^ prefixes

  tbnames = [
    'es_ancora',
    'en_ewt',
    'cs_pdt',
    'tr_imst',
    'ar_padt',
    'de_gsd',
  ]

  template_in = "data/%s/%s"
  template_out = "data/%s/%s.%s.conllu.%s"

  for tb in tbnames:
    if tb != "es_ancora": continue
    print(":: ",tb)
    # build filenames
    exp_args = args
    exp_args.train_file = template_in % (tb,"train")
    exp_args.dev_file = template_in % (tb,"dev")
    #
    train_fn = template_out % (tb,"train",args.src_ref,"gold")
    gold_fn = template_out % (tb,"dev",args.src_ref,"gold")
    src_fn  = template_out % (tb,"dev",args.src_ref,"pred")
    tgt_fn  = template_out % (tb,"dev",args.tgt_ref,"pred")

    loader = DataLoaderAnalizer(exp_args)
    train = loader.load_data("train")
    dev   = loader.load_data("dev")

    train_tups = get_form_lemmas(train_fn)
    gold_tups = get_form_lemmas(gold_fn)
    src_tups = get_form_lemmas(src_fn)
    tgt_tups = get_form_lemmas(tgt_fn)

    train_mapper = get_form_lemma_mapper(train_tups)
    gold_mapper = get_form_lemma_mapper(gold_tups)
    src_mapper = get_form_lemma_mapper(src_tups)
    tgt_mapper = get_form_lemma_mapper(tgt_tups)

    joint_keys = set(list(train_mapper.keys()) + list(gold_mapper.keys()))
    joint_map = {x:train_mapper[x] | gold_mapper[x] for x in joint_keys}

    pdb.set_trace()

    # ambiguous
    amb_forms = set([x for x,y in joint_map.items() if len(y)>1])
    src_amb_acc = get_custom_acc(gold_tups,src_tups,amb_forms)
    tgt_amb_acc = get_custom_acc(gold_tups,tgt_tups,amb_forms)

    pdb.set_trace()

    # unseen
    unk_id = loader.vocab_oplabel.get_label_id(UNK_TOKEN)
    unseen_forms = set([x for x,y in gold_mapper.items() if loader.vocab_oplabel.get_label_id(reformat_action(x))==unk_id ])
    src_uns_acc = get_custom_acc(gold_tups,src_tups,unseen_forms)
    tgt_uns_acc = get_custom_acc(gold_tups,tgt_tups,unseen_forms)

    # seen unambiguous
    seen_unamb_forms = set([x for x,y in gold_mapper.items() if len(y)==1 and x not in unseen_forms])
    src_su_acc = get_custom_acc(gold_tups,src_tups,seen_unamb_forms)
    tgt_su_acc = get_custom_acc(gold_tups,tgt_tups,seen_unamb_forms)

    print(src_amb_acc,tgt_amb_acc)
    print(src_uns_acc,tgt_uns_acc)
    print(src_su_acc,tgt_su_acc)


    pdb.set_trace()

    for w,lems in gold_mapper.items():
      if len(lems)<2: continue

