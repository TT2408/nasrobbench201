#!/bin/sh

values=`cat NasBenchID_6466_idx.txt`
echo "$values"
for archidx in $values; do
    echo $val
    python main.py --data cifar10 --epochs 50 --archidx $archidx --seed 1
done