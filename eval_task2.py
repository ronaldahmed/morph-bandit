import os
import glob as gb
import argparse
from basic_evaluator import BasicEvaluator

import pdb

def read_col(infile,col=2,lower=True):
  """ extracts column col form CONLLU file """
  for line in open(infile,'r'):
    line = line.strip("\n")
    if line=="" or line.startswith("# "): continue
    cols = line.split("\t")
    if lower:
      yield cols[col].lower()
    else:
      yield cols[col]


if __name__=="__main__":
  parser = argparse.ArgumentParser() 
  parser.add_argument("--gold", "-g", type=str, help="Gold conllu file")
  parser.add_argument("--pred", "-p", type=str, help="Pred conllu file")
  parser.add_argument("--lang", "-l", type=str, default="-", help="Language id")
  parser.add_argument("--baseline", "-b", type=str, default="lemming", help="Baseline code [lemming,neural]")
  parser.add_argument("--all","-a", action='store_true', help="Evaluate all langs")

  
  args = parser.parse_args()
  evaluator = BasicEvaluator()

  if not args.all:
    gold_gen = read_col(args.gold,2) # read lemma
    pred_gen = read_col(args.pred,2) # read lemma

    metrics = evaluator.eval_lemma_all(gold_gen,pred_gen)

    for mr in metrics:
      print("%s=%.4f" % (mr.desc,mr.res))

  else:
    with open("eval-task2-%s.csv" % args.baseline,'w') as outfile:
      print("treebank,lem-acc,lem-dist,msd-acc,msd-f1", file=outfile)
      dirnames = gb.glob("task2/UD_*")
      dirnames.sort()
      for dirname in dirnames:
        #if "Maltese" not in dirname: continue

        tb_dir = os.path.basename(dirname)
        pred_dev_files = gb.glob(os.path.join(dirname,"*pred-%s-dev.conllu" % args.baseline))
        if len(pred_dev_files)<1:
          continue
        gold_file = gb.glob(os.path.join(dirname,"*um-dev.conllu"))[0]
        pred_file = pred_dev_files[0]
        
        # print("files: ",gold_file,pred_file)
        # pdb.set_trace()

        gold_gen = read_col(gold_file,2) # read lemma
        pred_gen = read_col(pred_file,2) # 
        lem_metrics = evaluator.eval_lemma_all(gold_gen,pred_gen)

        gold_gen = read_col(gold_file,5) # read feats
        pred_gen = read_col(pred_file,5) # 
        msd_metrics = evaluator.eval_msd_all(gold_gen,pred_gen)
        
        res_lem = ["%.2f" % mr.res for mr in lem_metrics] # lem-acc, lem-dist
        res_msd = ["%.2f" % mr.res for mr in msd_metrics] # msd acc & f1
        print("%s,%s" % (tb_dir,",".join(res_lem + res_msd)), file=outfile)
        print("%s,%s" % (tb_dir,",".join(res_lem + res_msd)))

        #pdb.set_trace()


