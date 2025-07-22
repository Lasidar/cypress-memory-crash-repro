#!/usr/bin/env python3
"""
Example test runner for VR disparity detection algorithms
"""

import sys
import numpy as np
from typing import Dict
import pytest
from test_vr_disparity_utils import (
    create_synthetic_stereo_pair,
    calculate_disparity_metrics,
    visualize_disparity_map,
    VRDisplayParams,
    simulate_vr_viewing_discomfort
)


class ExampleDisparityDetector:
    """
    Example implementation of a disparity detector for testing
    This is a simple block matching algorithm - replace with your actual implementation
    """
    
    def __init__(self, block_size: int = 15, max_disparity: int = 64):
        self.block_size = block_size
        self.max_disparity = max_disparity
    
    def detect_disparity(self, left_image: np.ndarray, right_image: np.ndarray) -> Dict:
        """
        Simple block matching disparity detection
        
        Args:
            left_image: Left eye view
            right_image: Right eye view
            
        Returns:
            Dictionary with disparity information
        """
        import cv2
        
        # Convert to grayscale
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Simple stereo block matching
        stereo = cv2.StereoBM_create(
            numDisparities=self.max_disparity,
            blockSize=self.block_size
        )
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray)
        
        # Normalize to uint8 range
        disparity_normalized = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Calculate additional metrics
        valid_mask = disparity > 0
        
        # Detect vertical disparity (simplified)
        vertical_disparity_detected = False
        vertical_disparity_pixels = 0
        
        # Simple vertical disparity check using gradient
        vert_gradient = np.gradient(disparity_normalized, axis=0)
        if np.max(np.abs(vert_gradient)) > 10:
            vertical_disparity_detected = True
            vertical_disparity_pixels = np.sum(np.abs(vert_gradient) > 5)
        
        # Calculate basic statistics
        if np.any(valid_mask):
            valid_disparities = disparity_normalized[valid_mask]
            max_disp = float(np.max(valid_disparities))
            mean_disp = float(np.mean(valid_disparities))
            std_disp = float(np.std(valid_disparities))
        else:
            max_disp = mean_disp = std_disp = 0.0
        
        # Simple confidence map based on texture
        confidence_map = self._calculate_confidence(left_gray)
        
        # Create occlusion mask (simplified)
        occlusion_mask = disparity <= 0
        
        # Calculate comfort score (simplified)
        comfort_score = 1.0 - min(max_disp / 255.0, 1.0)
        
        return {
            "disparity_map": disparity_normalized,
            "max_disparity": max_disp,
            "mean_disparity": mean_disp,
            "std_disparity": std_disp,
            "bad_pixel_percentage": 5.0,  # Placeholder
            "mse": 100.0,  # Placeholder
            "ssim_score": 0.85,  # Placeholder
            "comfort_score": comfort_score,
            "confidence_map": confidence_map,
            "occlusion_mask": occlusion_mask,
            "vertical_disparity_detected": vertical_disparity_detected,
            "vertical_disparity_pixels": int(vertical_disparity_pixels),
            "vergence_conflict_score": 0.3,  # Placeholder
            "scale_mismatch": 0.1,  # Placeholder
            "rotation_mismatch": 0.05,  # Placeholder
            "color_mismatch": 0.2,  # Placeholder
            "sharpness_mismatch": 0.15,  # Placeholder
            "depth_continuity_score": 0.8,  # Placeholder
            "processing_time": 0.025  # Placeholder
        }
    
    def _calculate_confidence(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate confidence based on local texture"""
        import cv2
        
        # Use Sobel gradients as texture measure
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude of gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1
        confidence = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        return confidence.astype(np.float32)


def run_example_tests():
    """Run example tests with the sample detector"""
    
    print("VR Disparity Detection Test Example")
    print("=" * 50)
    
    # Create example detector
    detector = ExampleDisparityDetector()
    
    # Generate test data
    print("\n1. Generating synthetic stereo pair...")
    left, right = create_synthetic_stereo_pair(
        480, 640, disparity=20, pattern="checkerboard"
    )
    
    # Run detection
    print("2. Running disparity detection...")
    result = detector.detect_disparity(left, right)
    
    # Display results
    print("\n3. Detection Results:")
    print(f"   - Max disparity: {result['max_disparity']:.1f} pixels")
    print(f"   - Mean disparity: {result['mean_disparity']:.1f} pixels")
    print(f"   - Comfort score: {result['comfort_score']:.2f}")
    print(f"   - Vertical disparity detected: {result['vertical_disparity_detected']}")
    
    # Calculate VR viewing discomfort
    print("\n4. VR Viewing Comfort Analysis:")
    display_params = VRDisplayParams(
        resolution=(1080, 1920),
        pixel_pitch=0.1,
        viewing_distance=50
    )
    
    discomfort = simulate_vr_viewing_discomfort(
        result['disparity_map'], display_params
    )
    
    print(f"   - Max angular disparity: {discomfort['max_angular_disparity']:.1f} arcmin")
    print(f"   - Overall comfort score: {discomfort['comfort_score']:.2f}")
    
    # Visualize results
    print("\n5. Saving visualization...")
    visualize_disparity_map(
        result['disparity_map'],
        "Example Disparity Detection Results",
        save_path="example_disparity_result.png"
    )
    
    print("\nVisualization saved to: example_disparity_result.png")


def run_pytest_with_custom_detector():
    """Run pytest with custom detector"""
    
    print("\nRunning pytest with example detector...")
    print("=" * 50)
    
    # Create a custom conftest.py content
    conftest_content = '''
import pytest
from run_vr_disparity_tests import ExampleDisparityDetector

@pytest.fixture
def disparity_detector():
    """Provide the example disparity detector for testing"""
    detector = ExampleDisparityDetector()
    return detector.detect_disparity
'''
    
    # Write temporary conftest.py
    with open('conftest_temp.py', 'w') as f:
        f.write(conftest_content)
    
    # Run pytest
    pytest_args = [
        'test_vr_disparity_detection.py',
        '-v',
        '--tb=short',
        '-k', 'not performance'  # Skip performance tests for example
    ]
    
    pytest.main(pytest_args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VR Disparity Detection Test Runner")
    parser.add_argument(
        '--mode', 
        choices=['example', 'pytest', 'both'],
        default='example',
        help='Run mode: example only, pytest only, or both'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['example', 'both']:
        run_example_tests()
    
    if args.mode in ['pytest', 'both']:
        run_pytest_with_custom_detector()
    
    print("\nDone!")