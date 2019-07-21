#!/bin/bash

infile=$1

mv $infile "$infile".tmp

python fix_pos.py "$infile".tmp $infile