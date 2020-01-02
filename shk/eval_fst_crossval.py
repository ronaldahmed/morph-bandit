import os,sys
import subprocess as sp
from morph_mapper import MorphMapper
import argparse
import pdb


def get_morph_annon(form,fst_file):
  parsed = sp.run(["flookup",fst_file],
                  stdout=sp.PIPE,
                  input=form.encode("utf8") )
  parsed = parsed.stdout.decode("utf8").strip('\n').split('\n')
  options = set()
  for option_line in parsed:
    inp,m_out = option_line.split('\t')
    if m_out=="+?":
      return []
    morph_tag = []
    m_out = m_out.split(" ")
    idx = m_out[1].find("[")
    root = m_out[1][:idx]
    options.add(root)
  options = list(options)
  options.sort(key=lambda x: len(x), reverse=True)

  if len(options)>0:
    return options[0]
  return form

        



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", '-i', default="", type=str, help="Treebank file in CONLLU format")
  parser.add_argument("--fst", '-fst', default=None, type=str, help="Compiled Morphological Analyser in fst format")
  parser.add_argument("--morph_map", '-m', default="morph_map_table.tsv", type=str, help="Morphological dictionary")
  parser.add_argument("--allom", '-a', default="allomorphs.tsv", type=str, help="Allomorphs dictionary")
  args = parser.parse_args()

  n_folds = 10
  template = "../data/shk_cuni/shk_cuni-ud-test-%d.conllu"
  for k in range(n_folds):
    fn = template % k
    outfile = open(fn+"_fst",'w')
    for line in open(fn,'r'):
      line = line.strip("\n")
      if line=="":
        print("",file=outfile)
        continue
      cols = line.split("\t")
      form = cols[1]
      pred = get_morph_annon(form,args.fst)

      cols[2] = pred
      print(*cols,sep="\t",file=outfile)
    outfile.close()

    pobj = sp.run(["python3","../2019/evaluation/evaluate_2019_task2.py",
                     "--reference", fn,
                     "--output"   , fn + "_fst",
                    ], capture_output=True)
    output_res = pobj.stdout.decode().strip("\n").strip(" ").split("\t")  
    output_res = [float(x) for x in output_res]

    print("Fold",k)
    print(output_res)
    
    



