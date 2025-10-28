#!/bin/bash

# Quick test script to verify the fixes

echo "=== Testing Mask Dimension Fix ==="
echo "Running test_fixes.py..."
python test_fixes.py

echo ""
echo "=== Testing GPU Selection Fix ==="
echo "Testing with different GPU selections..."

# Test with GPU 0
echo "Testing --gpus 0:"
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_gpu0 --mask_mode m4 --gpus 0 2>&1 | grep -E "(Original GPU|Remapped GPU|device:|CUDA_VISIBLE)"

echo ""
echo "Testing --gpus 1:"
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_gpu1 --mask_mode m4 --gpus 1 2>&1 | grep -E "(Original GPU|Remapped GPU|device:|CUDA_VISIBLE)"

echo ""
echo "Testing --gpus 2:"
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_gpu2 --mask_mode m4 --gpus 2 2>&1 | grep -E "(Original GPU|Remapped GPU|device:|CUDA_VISIBLE)"

echo ""
echo "=== Testing Mask Mode m4 (Post-mask) ==="
echo "This should not crash with dimension mismatch error..."
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 1 --job_dir /tmp/test_m4 --mask_mode m4 --gpus 0 2>&1 | tail -10

echo ""
echo "All tests completed!"
