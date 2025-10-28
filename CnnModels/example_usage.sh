#!/bin/bash

# Example usage of CnnModels with wandb and bimask modes

# Train ResNet-32 on CIFAR-10 with m2 mode (bidirectional mask) and wandb logging
echo "Training ResNet-32 on CIFAR-10 with m2 mode and wandb..."
python cifar.py \
    --arch resnet32_cifar10 \
    --lr 0.1 \
    --weight_decay 0.001 \
    --data_path PATH_TO_DATASETS \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar_resnet32_m2 \
    --mask_mode m2 \
    --wandb_project bimask_cnn \
    --wandb_name resnet32_cifar10_m2

# Train ResNet-32 on CIFAR-10 with m3 mode (forward mask only) and wandb logging
echo "Training ResNet-32 on CIFAR-10 with m3 mode and wandb..."
python cifar.py \
    --arch resnet32_cifar10 \
    --lr 0.1 \
    --weight_decay 0.001 \
    --data_path PATH_TO_DATASETS \
    --label_smoothing 0.1 \
    --num_epochs 300 \
    --job_dir experiments/cifar_resnet32_m3 \
    --mask_mode m3 \
    --wandb_project bimask_cnn \
    --wandb_name resnet32_cifar10_m3

# Train ResNet-50 on ImageNet with m2 mode and wandb logging
echo "Training ResNet-50 on ImageNet with m2 mode and wandb..."
python imagenet.py \
    --arch resnet50 \
    --lr 0.1 \
    --data_path PATH_TO_DATASETS \
    --label_smoothing 0.1 \
    --num_epochs 120 \
    --job_dir experiments/imagenet_resnet50_m2 \
    --iter 100 \
    --greedy_num 100 \
    --mask_mode m2 \
    --wandb_project bimask_cnn \
    --wandb_name resnet50_imagenet_m2

# Train ResNet-50 on ImageNet with m3 mode and wandb logging
echo "Training ResNet-50 on ImageNet with m3 mode and wandb..."
python imagenet.py \
    --arch resnet50 \
    --lr 0.1 \
    --data_path PATH_TO_DATASETS \
    --label_smoothing 0.1 \
    --num_epochs 120 \
    --job_dir experiments/imagenet_resnet50_m3 \
    --iter 100 \
    --greedy_num 100 \
    --mask_mode m3 \
    --wandb_project bimask_cnn \
    --wandb_name resnet50_imagenet_m3

echo "All training examples completed!"
