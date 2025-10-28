#!/bin/bash

# Example usage of Random Mask functionality

echo "=== Random Mask Training Examples ==="
echo ""

echo "1. Training ResNet-32 on CIFAR-10 with random mask (50% sparsity):"
echo "python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 300 --job_dir experiments/cifar_resnet32_random --use_random_mask --random_mask_ratio 0.5 --wandb_project bimask_cnn --wandb_name resnet32_random_50"

echo ""
echo "2. Training ResNet-32 on CIFAR-10 with random mask (30% sparsity):"
echo "python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 300 --job_dir experiments/cifar_resnet32_random_30 --use_random_mask --random_mask_ratio 0.3 --wandb_project bimask_cnn --wandb_name resnet32_random_30"

echo ""
echo "3. Training ResNet-50 on ImageNet with random mask (50% sparsity):"
echo "python imagenet.py --arch resnet50 --lr 0.1 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 120 --job_dir experiments/imagenet_resnet50_random --use_random_mask --random_mask_ratio 0.5 --wandb_project bimask_cnn --wandb_name resnet50_random_50"

echo ""
echo "4. Training DeiT-small on ImageNet with random mask (50% sparsity):"
echo "python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir experiments/deit_random --use_random_mask --random_mask_ratio 0.5 --wandb_project bimask_cnn --wandb_name deit_random_50"

echo ""
echo "5. Comparison: N:M vs Random mask"
echo "N:M mask (2:4 pattern):"
echo "python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 300 --job_dir experiments/cifar_resnet32_nm --mask_mode m2 --wandb_project bimask_cnn --wandb_name resnet32_nm_2_4"

echo ""
echo "Random mask (50% sparsity):"
echo "python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 300 --job_dir experiments/cifar_resnet32_random --use_random_mask --random_mask_ratio 0.5 --wandb_project bimask_cnn --wandb_name resnet32_random_50"

echo ""
echo "=== Random Mask Features ==="
echo "- Use --use_random_mask flag to enable random mask"
echo "- Use --random_mask_ratio to set sparsity ratio (default: 0.5 for 50%)"
echo "- Random mask uses topk algorithm based on absolute weight values"
echo "- Compatible with all mask modes (m2, m3, m4)"
echo "- Works with both CNN and Vision Transformer models"
