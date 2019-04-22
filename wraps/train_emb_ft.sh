#!/bin/bash

treebank=$1
fasttext skipgram -input train-ops/$treebank -output emb/$treebank
