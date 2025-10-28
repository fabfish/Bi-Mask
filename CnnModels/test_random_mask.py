#!/usr/bin/env python3
"""
Test script to verify random mask functionality
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.options import args
from utils.conv_type import NMConv, get_random_sparse_matrix, get_n_m_sparse_matrix

def test_random_mask_function():
    """Test random mask function"""
    print("=== Testing Random Mask Function ===")
    
    # Create a test weight tensor
    w = torch.randn(64, 27)  # 64 output channels, 27 input features (3x3x3)
    
    print(f"Original weight shape: {w.shape}")
    print(f"Original weight sparsity: {(w == 0).float().mean():.3f}")
    
    # Test random mask with 50% sparsity
    w_sparse, mask = get_random_sparse_matrix(w, ratio=0.5)
    actual_sparsity = (mask == 0).float().mean()
    
    print(f"Random mask (50%): actual sparsity = {actual_sparsity:.3f}")
    print(f"Mask shape: {mask.shape}")
    print(f"Sparse weight shape: {w_sparse.shape}")
    
    # Test random mask with 30% sparsity
    w_sparse_30, mask_30 = get_random_sparse_matrix(w, ratio=0.3)
    actual_sparsity_30 = (mask_30 == 0).float().mean()
    
    print(f"Random mask (30%): actual sparsity = {actual_sparsity_30:.3f}")
    
    # Test N:M mask for comparison
    w_nm, mask_nm = get_n_m_sparse_matrix(w)
    actual_sparsity_nm = (mask_nm == 0).float().mean()
    
    print(f"N:M mask (2:4): actual sparsity = {actual_sparsity_nm:.3f}")

def test_nmconv_random_mask():
    """Test NMConv with random mask"""
    print("\n=== Testing NMConv with Random Mask ===")
    
    # Test different configurations
    test_configs = [
        {"use_random_mask": False, "random_mask_ratio": 0.5, "desc": "N:M mask (2:4)"},
        {"use_random_mask": True, "random_mask_ratio": 0.5, "desc": "Random mask (50%)"},
        {"use_random_mask": True, "random_mask_ratio": 0.3, "desc": "Random mask (30%)"},
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['desc']}")
        
        # Temporarily set args
        original_use_random = args.use_random_mask
        original_ratio = args.random_mask_ratio
        
        args.use_random_mask = config["use_random_mask"]
        args.random_mask_ratio = config["random_mask_ratio"]
        
        try:
            # Create NMConv layer
            conv = NMConv(3, 64, 3, padding=1)
            
            print(f"  use_random_mask: {conv.use_random_mask}")
            print(f"  random_mask_ratio: {conv.random_mask_ratio}")
            print(f"  mask_mode: {conv.mask_mode}")
            
            # Test forward pass
            x = torch.randn(1, 3, 32, 32)
            output = conv(x)
            
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Forward mask shape: {conv.forward_mask.shape}")
            
            # Calculate actual sparsity
            if conv.use_random_mask:
                actual_sparsity = (conv.forward_mask == 0).float().mean()
                print(f"  Actual sparsity: {actual_sparsity:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
        finally:
            # Restore original values
            args.use_random_mask = original_use_random
            args.random_mask_ratio = original_ratio

if __name__ == "__main__":
    test_random_mask_function()
    test_nmconv_random_mask()
