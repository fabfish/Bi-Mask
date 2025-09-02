#!/bin/bash
# use tee to log output
cd CnnModels
python cifar.py --arch resnet32_cifar10 --gpus 0 1 2 3 --lr 0.1 --weight_decay 0.001 --data_path /data/cifar10 --label_smoothing 0.1 --num_epochs 300 --job_dir /data/yuzhiyuan 2>&1 | tee resnet32_cifar10.log