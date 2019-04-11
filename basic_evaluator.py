import numpy as np
from label_dictionary import LabelDictionary
from collections import namedtuple
from sklearn.metrics import f1_score, accuracy_score
import re
import warnings

import pdb


warnings.filterwarnings("ignore")
Eval = namedtuple('Eval', 'desc long_desc res')
EMPTY_ENTRY_ = re.compile(r'^_(?:;_)*$')

class BasicEvaluator(object):
  def __init__(self):
    self.label_dict = LabelDictionary()


  def evaluate_lemma_acc_dist(self, ground_truth, predict):
    '''
    evaluate single instance
    '''
    correct = 1.0 * (predict == ground_truth)
    dist = edit_distance(predict, ground_truth)
    return correct, dist


  def eval_lemma_all(self, gold_data, pred_data):
    '''
    evaluate all instances
    @param gold_data: [w1,w2,...] # no sent ordering, just list of words
    '''
    # if len(gold_data) != len(pred_data):
    #   print("-> |gold data| != |pred data| !! ")
    #   pdb.set_trace()

    correct, distance = 0, 0
    ntotal = 0
    for gd, pd in zip(gold_data,pred_data):
      corr, dist = self.evaluate_lemma_acc_dist(gd, pd)
      correct += corr
      distance += dist
      ntotal += 1
      # if ntotal % 5 == 0:
      #   pdb.set_trace()

    acc = round(correct / ntotal * 100, 4)
    distance = round(distance / ntotal, 4)
    return [
      Eval('lem-acc', 'lemmata accuracy', acc),
      Eval('lem-dist', 'average edit distance', distance)
    ]


  def update_label_dict(self,feat):
    return set([ self.label_dict.add(lab) for lab in feat.split(";")])


  def resort_label(self,label):
    labs = label.split(";")
    labs.sort() # lexicographic order
    return ";".join(labs)


  def eval_msd_all(self,gold_data, pred_data):
    """ Details:
          - no empty lines considered ("_")
          - pre - sorting labels before comparison (works as upperbound on official scores)
          - F1 score as weighted instead of micro in label-score aggregation
          -   (different from dataset-aggregation)
    """

    correct, ntotal = 0,0
    gold,pred = [],[]
    for gd, pd in zip(gold_data,pred_data):
      gd = self.resort_label(gd)
      pd = self.resort_label(pd)
      gold.append(gd)
      pred.append(pd)
    gold = np.array(gold)
    pred = np.array(pred)

    #pdb.set_trace()

    not_empty_masker = np.vectorize(lambda p: not EMPTY_ENTRY_.match(p))
    criterion = not_empty_masker(gold)

    return [
      Eval('msd-acc', 'msd accuracy', 100.0*accuracy_score(gold[criterion],pred[criterion])),
      # Eval('msd-f1' , 'msd F1 score', 100.0*f1_score      (gold[criterion],pred[criterion],average="micro"))
      Eval('msd-f1' , 'msd F1 score', 100.0*f1_score      (gold[criterion],pred[criterion],average="weighted"))
    ]





def edit_distance(str1, str2):
  '''Simple Levenshtein implementation for evalm.'''
  table = np.zeros([len(str2) + 1, len(str1) + 1])
  for i in range(1, len(str2) + 1):
    table[i][0] = table[i - 1][0] + 1
  for j in range(1, len(str1) + 1):
    table[0][j] = table[0][j - 1] + 1
  for i in range(1, len(str2) + 1):
    for j in range(1, len(str1) + 1):
      if str1[j - 1] == str2[i - 1]:
        dg = 0
      else:
        dg = 1
      table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                table[i - 1][j - 1] + dg)
  return int(table[len(str2)][len(str1)])


