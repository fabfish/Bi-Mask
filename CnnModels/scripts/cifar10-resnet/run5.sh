#!/bin/bash

# Train ResNet-32 on CIFAR-10 with m3 mode and wandb logging
echo "Training ResNet-32 on CIFAR-10 with m2 mode and wandb..."
CUDA_VISIBLE_DEVICES=1 python3 cifar.py \
    --arch vgg19_cifar100 \
    --lr 0.1 \
    --gpus 1 \
    --weight_decay 0.001 \
    --data_path /home/yzy/GitHub/Bi-Mask/CnnModels/datasets \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar_vgg19_m1 \
    --mask_mode m1 \
    --wandb_project bimask_cnn_vgg_cifar100 \
    --nm_layers layer3 \
    --seed 24 \
    --wandb_name vgg19_cifar100_m1_rand31_l3_sd24
