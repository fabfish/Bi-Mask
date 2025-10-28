#!/usr/bin/env python3
"""
Test script to verify mask dimension fix and GPU selection
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.options import args
from utils.conv_type import NMConv

def test_mask_dimensions():
    """Test mask dimension compatibility"""
    print("=== Testing Mask Dimensions ===")
    
    # Create a test NMConv layer
    conv = NMConv(3, 64, 3, padding=1)
    
    print(f"Original weight shape: {conv.weight.shape}")
    print(f"Forward mask shape: {conv.forward_mask.shape}")
    
    # Test post_mask_apply
    try:
        conv.post_mask_apply()
        print("✅ post_mask_apply() executed successfully")
        print(f"Updated weight shape: {conv.weight.shape}")
    except Exception as e:
        print(f"❌ post_mask_apply() failed: {e}")

def test_gpu_selection():
    """Test GPU selection logic"""
    print("\n=== Testing GPU Selection ===")
    
    # Simulate different GPU selections
    test_gpu_selections = [[0], [1], [2], [0, 1], [1, 2]]
    
    for gpu_list in test_gpu_selections:
        print(f"\nTesting GPU selection: {gpu_list}")
        
        # Simulate the GPU selection logic
        visible_gpus_str = ','.join(str(i) for i in gpu_list)
        original_gpus = gpu_list.copy()
        remapped_gpus = [i for i in range(len(gpu_list))]
        
        print(f"  Original GPU selection: {original_gpus}")
        print(f"  CUDA_VISIBLE_DEVICES: {visible_gpus_str}")
        print(f"  Remapped GPU indices: {remapped_gpus}")
        
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{remapped_gpus[0]}")
            print(f"  Device: {device}")
        else:
            print("  CUDA not available")

if __name__ == "__main__":
    test_mask_dimensions()
    test_gpu_selection()
