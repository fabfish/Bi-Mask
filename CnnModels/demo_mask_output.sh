#!/bin/bash

# Demo script to show mask mode output functionality

echo "=== Mask Mode Output Demo ==="
echo "This script demonstrates the mask mode configuration output"
echo ""

echo "1. Testing CIFAR-10 ResNet-32 with different mask modes:"
echo ""

echo "Testing m2 mode (bidirectional mask):"
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_m2 --mask_mode m2 --wandb_project test 2>&1 | grep -A 20 "Model Mask Mode Configuration"

echo ""
echo "Testing m3 mode (pre-mask):"
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_m3 --mask_mode m3 --wandb_project test 2>&1 | grep -A 20 "Model Mask Mode Configuration"

echo ""
echo "Testing m4 mode (post-mask):"
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_m4 --mask_mode m4 --wandb_project test 2>&1 | grep -A 20 "Model Mask Mode Configuration"

echo ""
echo "Demo completed!"
