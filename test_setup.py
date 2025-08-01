#!/usr/bin/env python3
"""
Test script to verify HRM setup with Intel Arc GPU support.
"""

import torch
import os

def test_pytorch_xpu():
    """Test PyTorch XPU (Intel Arc) support."""
    print("=== PyTorch XPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {torch.xpu.is_available()}")
    
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        print(f"Current XPU device: {torch.xpu.current_device()}")
        
        # Test tensor operations on XPU
        x = torch.randn(5, 3, device='xpu')
        y = torch.randn(3, 5, device='xpu')
        z = torch.mm(x, y)
        print(f"XPU tensor operations working: {z.shape}")
        return True
    else:
        print("XPU not available - falling back to CPU")
        return False

def test_adam_atan2():
    """Test AdamATan2 fallback optimizer."""
    print("\n=== AdamATan2 Test ===")
    try:
        from adam_atan2 import AdamATan2
        
        # Simple model for testing
        device = 'xpu' if torch.xpu.is_available() else 'cpu'
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = AdamATan2(model.parameters(), lr=0.01)
        
        # Test optimization step
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)
        
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
        
        print(f"AdamATan2 optimizer working - final loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"AdamATan2 test failed: {e}")
        return False

def test_basic_imports():
    """Test basic project imports."""
    print("\n=== Import Test ===")
    try:
        import puzzle_dataset
        from utils.functions import load_model_class
        print("Core imports successful")
        return True
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

def main():
    """Run all setup tests."""
    print("HRM Setup Verification Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_pytorch_xpu():
        tests_passed += 1
    
    if test_adam_atan2():
        tests_passed += 1
        
    if test_basic_imports():
        tests_passed += 1
    
    print(f"\n=== Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ Setup verification successful!")
        print("\nYour HRM environment is ready with Intel Arc GPU support.")
        print("You can now run the quick Sudoku demo or other experiments.")
        print("\nNote: FlashAttention was skipped (not compatible with Intel Arc).")
        print("The model will use standard attention mechanisms instead.")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
