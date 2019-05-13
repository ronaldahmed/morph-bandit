order_ = open("tbnames_order_tab.txt",'r').read().split("\n")
to_map = {}
for line in open("res.dev.csv"):
	line = line.strip("\n")
	if line=="": continue
	tb,acc,edist = line.split(",")
	to_map[tb] = (acc,edist)

outfile = open("ordered_res.dev.csv",'w')
for tb in order_:
	print("%s,%s,%s" % (tb,to_map[tb][0],to_map[tb][1]), file = outfile )
