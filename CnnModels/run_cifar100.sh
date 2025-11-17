#!/bin/bash
# Simple script to run CIFAR-100 training with reasonable defaults
python cifar.py \
  --dataset cifar100 \
  --data_path ./data \
  --arch resnet18 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --num_epochs 200 \
  --lr 0.1 \
  --gpus 0
