#!/bin/bash

# Train ResNet-32 on CIFAR-10 with m3 mode and wandb logging
echo "Training ResNet-32 on CIFAR-10 with m4 mode and wandb..."
CUDA_VISIBLE_DEVICES=2 python3 cifar.py \
    --arch resnet32_cifar10 \
    --lr 0.1 \
    --gpus 3 \
    --weight_decay 0.001 \
    --data_path /root/Bi-Mask/datasets \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar_resnet32_m4_fast \
    --mask_mode m4 \
    --use_random_mask \
    --random_mask_ratio 0.9 \
    --wandb_project bimask_cnn_random \
    --wandb_name resnet32_cifar10_m4_fast
