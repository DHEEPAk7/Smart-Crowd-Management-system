"""
check_gpu.py - GPU Status Checker
Run this to check if your GPU is properly set up for the application
"""

import torch
import sys

def check_gpu():
    """Check GPU availability and status"""
    print("=" * 60)
    print("GPU STATUS CHECK")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("\n✅ GPU DETECTED!")
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"   Number of GPUs: {num_gpus}")
        
        # Get details for each GPU
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"   - CUDA Capability: {props.major}.{props.minor}")
            total_mem = props.total_memory / 1e9
            print(f"   - Total Memory: {total_mem:.2f} GB")
            
            # Get allocated memory
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"   - Allocated Memory: {allocated:.2f} GB")
            print(f"   - Reserved Memory: {reserved:.2f} GB")
            print(f"   - Available Memory: {(total_mem - allocated):.2f} GB")
        
        # Test GPU
        print("\n🧪 Testing GPU computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   ✅ GPU computation test PASSED")
        except Exception as e:
            print(f"   ❌ GPU computation test FAILED: {e}")
        
        print("\n🚀 Ready to use GPU! Your program will run on GPU.")
        
    else:
        print("\n⚠️  NO GPU DETECTED")
        print("   Your program will run on CPU (much slower)")
        print("\n   To use GPU:")
        print("   1. Install NVIDIA GPU drivers")
        print("   2. Install CUDA toolkit (compatible with your PyTorch version)")
        print("   3. Install cuDNN (for optimized operations)")
        print("   4. Reinstall PyTorch with CUDA support:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION: Edit app.py to see GPU status on startup")
    print("=" * 60)

if __name__ == "__main__":
    check_gpu()
