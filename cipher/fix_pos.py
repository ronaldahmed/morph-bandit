import sys
from utils import um2char
import re

infn = sys.argv[1]
outfn = sys.argv[2]

re_pats = []
for umpos,ch in um2char.items():
	tmp = "".join(["["+x+"]" for x in umpos])
	pat = re.compile(r"\s" + tmp + r"\s",re.UNICODE)
	re_pats.append([pat,ch])

outfile = open(outfn,'w')
for line in open(infn,'r'):
	for pat,ch in re_pats:
		line = pat.sub(ch,line)
	print(line,file=outfile)

