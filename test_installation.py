#!/usr/bin/env python3
"""
Test script to verify the VR Video Quality Assessment tool installation.
"""

import sys
import importlib.util


def check_dependency(module_name, package_name=None):
    """Check if a Python module is installed."""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {package_name} is NOT installed")
        return False
    else:
        print(f"✅ {package_name} is installed")
        return True


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    print("-" * 40)
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ Failed to import NumPy")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ Failed to import OpenCV")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ Failed to import PyTorch")
        return False
    
    try:
        import pyiqa
        print(f"✅ PyIQA is installed")
        # List available metrics
        available_metrics = pyiqa.list_models()
        print(f"   Available metrics: {len(available_metrics)}")
    except ImportError:
        print("❌ Failed to import PyIQA")
        return False
    
    try:
        import tqdm
        print(f"✅ tqdm is installed")
    except ImportError:
        print("❌ Failed to import tqdm")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Failed to import Pandas")
        return False
    
    return True


def test_vr_assessor():
    """Test if the VR Video Quality Assessor can be initialized."""
    print("\nTesting VR Video Quality Assessor...")
    print("-" * 40)
    
    try:
        from vr_video_quality_assessment import VRVideoQualityAssessor
        print("✅ Successfully imported VRVideoQualityAssessor")
        
        # Try to create an instance
        print("Creating assessor instance...")
        assessor = VRVideoQualityAssessor(device='cpu')  # Use CPU for testing
        print(f"✅ Assessor created successfully")
        print(f"   Initialized metrics: {len(assessor.metrics)}")
        
        # List initialized metrics
        if assessor.metrics:
            print("   Available metrics:")
            for metric_name in sorted(assessor.metrics.keys()):
                print(f"     - {metric_name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import VRVideoQualityAssessor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating assessor: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with a dummy image."""
    print("\nTesting basic functionality...")
    print("-" * 40)
    
    try:
        import numpy as np
        import torch
        from vr_video_quality_assessment import VRVideoQualityAssessor
        
        # Create a dummy image
        dummy_frame = np.random.randint(0, 255, (480, 960, 3), dtype=np.uint8)
        print("✅ Created dummy VR frame (480x960x3)")
        
        # Create assessor
        assessor = VRVideoQualityAssessor(device='cpu')
        
        # Process frame
        left_eye, right_eye = assessor.process_frame(dummy_frame, 'equirectangular')
        print(f"✅ Processed frame: left_eye shape={left_eye.shape}, right_eye shape={right_eye.shape}")
        
        # Test tensor conversion
        tensor = assessor._prepare_tensor(left_eye)
        print(f"✅ Tensor conversion successful: shape={tensor.shape}, device={tensor.device}")
        
        # Test one metric
        if 'psnr' in assessor.metrics:
            try:
                # Create a reference (slightly modified version)
                ref_frame = left_eye.copy()
                ref_frame = (ref_frame * 0.9).astype(np.uint8)
                
                left_tensor = assessor._prepare_tensor(left_eye)
                ref_tensor = assessor._prepare_tensor(ref_frame)
                
                score = assessor.metrics['psnr'](left_tensor, ref_tensor)
                print(f"✅ PSNR metric test: score={float(score.cpu().item()):.2f}")
            except Exception as e:
                print(f"⚠️  PSNR metric test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("VR Video Quality Assessment - Installation Test")
    print("=" * 60)
    print()
    
    all_tests_passed = True
    
    # Check dependencies
    print("Checking dependencies...")
    print("-" * 40)
    dependencies = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('torch', 'PyTorch'),
        ('pyiqa', 'PyIQA'),
        ('tqdm', 'tqdm'),
        ('pandas', 'Pandas')
    ]
    
    for module, name in dependencies:
        if not check_dependency(module, name):
            all_tests_passed = False
    
    print()
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test VR assessor
    if not test_vr_assessor():
        all_tests_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ All tests passed! The tool is ready to use.")
        print("\nTo process VR videos, run:")
        print("  python vr_video_quality_assessment.py /path/to/videos output.csv")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())