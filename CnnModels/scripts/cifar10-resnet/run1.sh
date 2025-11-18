#!/bin/bash

# Train ResNet-32 on CIFAR-10 with m3 mode and wandb logging
echo "Training ResNet-32 on CIFAR-10 with m3 mode and wandb..."
CUDA_VISIBLE_DEVICES=1 python3 cifar.py \
    --arch resnet32_cifar10 \
    --lr 0.1 \
    --gpus 1 \
    --weight_decay 0.001 \
    --data_path /root/Bi-Mask/datasets \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar_resnet32_m3 \
    --mask_mode m3 \
    --wandb_project bimask_cnn_random \
    --nm_layers layer3 \
    --seed 24 \
    --wandb_name resnet32_cifar10_m3_rand31_l3_sd24
