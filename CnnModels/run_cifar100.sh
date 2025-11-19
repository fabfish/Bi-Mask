#!/bin/bash
# Simple script to run CIFAR-100 training with reasonable defaults

echo "Training ResNet-32 on CIFAR-100 with m1 mode and wandb..."
CUDA_VISIBLE_DEVICES=0 python3 cifar.py \
    --arch vgg19_cifar100 \
    --dataset cifar100 \
    --lr 0.1 \
    --gpus 0 \
    --weight_decay 0.001 \
    --data_path /home/yzy/GitHub/Bi-Mask/datasets \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar100_vgg19_m1 \
    --mask_mode m1 \
    --wandb_project bimask_cnn_vgg19_cifar100_test \
    --nm_layers "features.30,features.33,features.36,features.40,features.43,features.46,features.49" \
    --seed 42 \
    --wandb_name vgg19_cifar100_m1
