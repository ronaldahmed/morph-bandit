#!/bin/bash

tbname=$1

tail -n+2 emb/$tbname.vec | tr ' ' '\t' | cut -f 1 --complement > emb/$tbname.vec.viz
tail -n+2 emb/$tbname.vec | cut -f 1 -d ' ' > emb/$tbname.vec.viz.meta


tail -n+2 emb/$tbname.vec | tr ' ' '\t' | cut -f 1 --complement > emb/$tbname.vec.viz
tail -n+2 emb/$tbname.vec | cut -f 1 -d ' ' > emb/$tbname.vec.viz.meta