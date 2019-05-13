order_ = open("tbnames_order_tab.txt",'r').read().split("\n")
to_map = {}
for line in open("res-anlz.dev.csv"):
	line = line.strip("\n")
	if line=="": continue
	tb,acc,edist,macc,mf1 = line.split(",")
	to_map[tb] = (acc,edist,macc,mf1)

outfile = open("ordered_res-anlz.dev.csv",'w')
for tb in order_:
	print("%s,%s" % (tb,",".join(to_map[tb]) ), file = outfile)
