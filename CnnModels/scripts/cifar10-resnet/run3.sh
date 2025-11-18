#!/bin/bash

# Train ResNet-32 on CIFAR-10 with m3 mode and wandb logging
echo "Training ResNet-32 on CIFAR-10 with m2 mode and wandb..."
CUDA_VISIBLE_DEVICES=3 python3 cifar.py \
    --arch resnet32_cifar100 \
    --lr 0.1 \
    --gpus 3 \
    --weight_decay 0.001 \
    --data_path /home/yzy/GitHub/Bi-Mask/CnnModels/datasets \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar_resnet32_m1 \
    --mask_mode m1 \
    --wandb_project bimask_cnn_random \
    --seed 24 \
    --wandb_name resnet32_cifar100_m1_rand31_l3_sd24
