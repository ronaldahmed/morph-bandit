import sys
import glob as gb
import pdb

START = "START"

for filename in gb.glob("dicts/*.txt"):
  outfile = open(filename[:-3] + "ops",'w')

  for line in open(filename,'r'):
    line = line.strip('\n')
    if line == '': continue
    try:
      w1,w2 = line.split('\t')
    except:
      w1,w2 = line.split(' ')
      

    opf1 = "START.START-" + w1.lower()
    opf2 = "START.START-" + w2.lower()
    print("%s\t%s" % (opf1,opf2), file=outfile)
  #
#
