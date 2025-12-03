#!/usr/bin/env python3
"""Test if PyTorch can use the GPU"""
import torch

print("=" * 60)
print("PyTorch GPU Compatibility Test")
print("=" * 60)

# Basic info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        cap = torch.cuda.get_device_capability(i)
        print(f"  Compute Capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test actual GPU computation
    print("\n" + "=" * 60)
    print("Testing GPU computation...")
    print("=" * 60)
    
    try:
        device = torch.device('cuda:0')
        print(f"\nUsing device: {device}")
        
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        print(f"✓ Created tensors on GPU")
        print(f"  x.device: {x.device}")
        print(f"  y.device: {y.device}")
        
        # Perform computation
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication successful!")
        print(f"  Result device: {z.device}")
        print(f"  Result shape: {z.shape}")
        print(f"  Result sum: {z.sum().item():.2f}")
        
        # Test backward pass (requires grad)
        x_grad = torch.randn(1000, 1000, device=device, requires_grad=True)
        y_grad = torch.randn(1000, 1000, device=device, requires_grad=True)
        z_grad = torch.matmul(x_grad, y_grad)
        z_grad.sum().backward()
        print(f"✓ Backward pass successful!")
        print(f"  x_grad.grad shape: {x_grad.grad.shape if x_grad.grad is not None else None}")
        
        print("\n" + "=" * 60)
        print("✓ GPU is WORKING - PyTorch can use the GPU!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ GPU computation FAILED: {e}")
        print("=" * 60)
        print("✗ GPU is NOT working properly")
        print("=" * 60)
else:
    print("\n✗ CUDA is not available")
    print("=" * 60)
    print("✗ GPU cannot be used")
    print("=" * 60)

