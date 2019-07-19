import sys
import pdb
import argparse
import unicodedata


# UniMorph  support
um2char = {
  "ADJ" :'A',
  "ADP" :"B",
  "ADV" :'C',
  "ART" : 'D',
  "AUX" :"E",
  "CLF" :'F',
  "COMP":'G',
  "CONJ":'H',
  "DET" :"I",
  "INTJ"  :'J',
  "N"  :'K',
  "NUM" :'L',
  "PART":'M',
  "PRO"  :"N",
  "PROPN" :'O',
  "V"  :"P",
  "V.CVB":"Q",
  "V.MSDR":"R",
  "V.PTCP":"S",
  "PUNCT":"T",
  "_":"U",
}


def test_punct(token):
  for c in token:
    if unicodedata.category(c)[0] != 'P':
      return False
  return True


def test_num(token):
  for c in token:
    if unicodedata.category(c)[0] != 'N':
      return False
  return True


def map_um_bundle_to_pos(token,bundle):
  # precedence of V subclasses over V,ADJ
  precedence = [
    "V.CVB",
    "V.MSDR",
    "V.PTCP",
  ]

  all_pos = list(um2char.keys())
  fine_grained = bundle.split(";")
  for f in fine_grained:
    if f in precedence:
      return f
  for f in fine_grained:
    if f in all_pos and f!="_":
      return f
  # special cases
  if test_punct(token):
    return "PUNCT"
  if test_num(token):
    return "NUM"

  return "_"
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input","-i", type=str, default=None, help="input conllu file")
  parser.add_argument("--mode" ,"-m", type=str, default="ch", help="Tag mode [ch,tag]")
  parser.add_argument("--col"  ,"-c", type=int, default=1, help="Column to extract [0-9]")
  parser.add_argument("--tb"   ,"-tb", type=str, default="ud", help="Treebank name [ud,ut]")
  parser.add_argument("--lid","-lid", action='store_true', help="Keep lang_id from text")

  args = parser.parse_args()

  text=""
  idx = args.col
  mode = args.mode

  count = 1

  for line in open(args.input,'r'):
    line = line.strip("\n")
    if line=='': continue
    cols = line.split('\t')
    if cols[0]=="1" and text!='':
      print(text.strip(' '))
      text=''
      count = 1
    token = ''
    datum = cols[idx]
    if idx==1 and not args.lid:
      datum = datum[:-3]

    if mode=='ch' and idx==5:
      datum = um2char[map_um_bundle_to_pos(cols[1],datum)]

    token = datum.strip(' ')

    all_dig = True
    for sw in token.split(' '):
      if not sw.isdigit():
        all_dig=False
        break
    token = token.replace(" ","") if all_dig else token.replace(" ","_")
    text += " "+token

    # print("|",token,"|")
    count += 1

  print(text.strip(' '))
