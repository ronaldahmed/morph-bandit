import sys

sent = []
for line in sys.stdin:
	line = line.strip("\n")
	if line=="":
		print(" ".join(sent))
		sent = []
	else:
		sent.append(line.lower())
