import sys
from utils import um2char
import re

infn = sys.argv[1]
outfn = sys.argv[2]

re_pats = []
umpos_ch = list(um2char.items())
umpos_ch.sort(key=lambda x: x[0], reverse=True)

for umpos,ch in umpos_ch:
	tmp = "".join(["["+x+"]" for x in umpos])
	pat = re.compile(r"\s" + tmp + r"\s",re.UNICODE)
	re_pats.append([pat,ch])

outfile = open(outfn,'w')
for line in open(infn,'r'):
	line = line.strip("\n")
	for pat,ch in re_pats:
		line = pat.sub(ch,line)
	print(line,file=outfile)

