#!/usr/bin/env python3
"""
Test script to demonstrate mask mode output functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.options import args
import models
import torch

def test_mask_mode_output():
    """Test function to show mask mode configuration output"""
    
    # Test different mask modes
    test_modes = ['m2', 'm3', 'm4']
    
    for mode in test_modes:
        print(f"\n{'='*60}")
        print(f"Testing with mask_mode: {mode}")
        print(f"{'='*60}")
        
        # Temporarily set the mask mode
        original_mode = args.mask_mode
        args.mask_mode = mode
        
        try:
            # Create model
            device = torch.device('cpu')  # Use CPU for testing
            model = models.__dict__['resnet32_cifar10']().to(device)
            
            # Print mask mode configuration
            print(f"\n=== Model Mask Mode Configuration ===")
            print(f"Global mask_mode: {args.mask_mode}")
            print(f"Layer-wise mask modes:")
            
            from utils.conv_type import NMConv
            layer_count = 0
            for name, module in model.named_modules():
                if isinstance(module, NMConv):
                    layer_count += 1
                    print(f"  Layer {layer_count}: {name} -> mask_mode: {module.mask_mode}")
            
            print(f"Total NMConv layers: {layer_count}")
            print("=" * 40)
            
        except Exception as e:
            print(f"Error testing mode {mode}: {e}")
        finally:
            # Restore original mode
            args.mask_mode = original_mode

if __name__ == "__main__":
    test_mask_mode_output()
