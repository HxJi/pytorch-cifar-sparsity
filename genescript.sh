#!/bin/bash
for i in {0..199}
do
  j=$[i+1]
  echo "python3 /home/hj14/pytorch-cifar/main.py  --resume --depth=$1 --ckpt=$i --train=0 | tee -a /shared/hj14/cifar10-dataset/sparsity-$1.log" >> script-$1.sh
done 