import sys

train,dev,test = sys.argv[1:]

train = set([x.lower() for x in open(train,"r")])
dev = set([x.lower() for x in open(dev,"r")])
test = set([x.lower() for x in open(test,"r")])

queries = train - dev - test

print("\n".join(queries))