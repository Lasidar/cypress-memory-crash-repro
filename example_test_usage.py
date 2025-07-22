"""
Example usage of VR disparity detection test suite.

This script demonstrates how to use the test suite to evaluate
a VR disparity detection algorithm.
"""

import numpy as np
import cv2
from typing import Dict
import pytest
import sys
import os

# Import test utilities
from test_vr_disparity_utils import (
    VRCameraParameters,
    AdvancedStereoGenerator,
    DisparityVisualization,
    VRComfortAnalyzer,
    generate_test_report
)

# Import the evaluator
from test_vr_disparity_detection import DisparityEvaluator


class ExampleDisparityAlgorithm:
    """
    Example implementation of a VR disparity detection algorithm.
    
    This is a simple block matching algorithm for demonstration.
    Replace this with your actual algorithm.
    """
    
    def __init__(self, block_size: int = 15, max_disparity: int = 64):
        """
        Initialize the algorithm.
        
        Args:
            block_size: Size of the matching block
            max_disparity: Maximum disparity to search
        """
        self.block_size = block_size
        self.max_disparity = max_disparity
        
        # Create OpenCV stereo matcher
        self.stereo = cv2.StereoBM_create(
            numDisparities=max_disparity,
            blockSize=block_size
        )
    
    def __call__(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from stereo pair.
        
        Args:
            left_image: Left eye image (H, W, 3) uint8
            right_image: Right eye image (H, W, 3) uint8
            
        Returns:
            Disparity map (H, W) uint8, normalized to 0-255
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Normalize to 0-255 range
        # OpenCV returns 16-bit fixed-point disparity (divide by 16)
        disparity = disparity.astype(np.float32) / 16.0
        
        # Clip negative values and normalize
        disparity[disparity < 0] = 0
        if disparity.max() > 0:
            disparity = (disparity / disparity.max() * 255).astype(np.uint8)
        else:
            disparity = disparity.astype(np.uint8)
        
        return disparity


def run_single_test():
    """Run a single test to demonstrate the evaluation process."""
    print("Running single test example...")
    
    # Create test data generator
    camera_params = VRCameraParameters()
    generator = AdvancedStereoGenerator(camera_params)
    
    # Generate a test scene
    left, right, ground_truth = generator.create_realistic_scene(
        scene_type="indoor",
        lighting="normal",
        add_noise=True
    )
    
    # Initialize algorithm
    algorithm = ExampleDisparityAlgorithm(block_size=15, max_disparity=64)
    
    # Compute disparity
    print("Computing disparity map...")
    disparity = algorithm(left, right)
    
    # Evaluate results
    evaluator = DisparityEvaluator()
    
    # Calculate metrics
    bad_pixel_error = evaluator.bad_pixel_error(
        disparity, ground_truth.disparity_map, threshold=3.0
    )
    mae = evaluator.mean_absolute_error(disparity, ground_truth.disparity_map)
    rmse = evaluator.root_mean_square_error(disparity, ground_truth.disparity_map)
    edge_score = evaluator.edge_preservation_score(disparity, ground_truth.disparity_map)
    ssim = evaluator.structural_similarity(disparity, ground_truth.disparity_map)
    
    print(f"\nEvaluation Results:")
    print(f"  Bad Pixel Error (3px): {bad_pixel_error:.2f}%")
    print(f"  Mean Absolute Error: {mae:.2f} pixels")
    print(f"  Root Mean Square Error: {rmse:.2f} pixels")
    print(f"  Edge Preservation Score: {edge_score:.3f}")
    print(f"  Structural Similarity: {ssim:.3f}")
    
    # Analyze VR comfort
    comfort_analyzer = VRComfortAnalyzer(camera_params)
    comfort_metrics = comfort_analyzer.analyze_comfort(disparity)
    
    print(f"\nVR Comfort Analysis:")
    print(f"  Max Vergence Angle: {comfort_metrics['max_vergence_angle_deg']:.2f}°")
    print(f"  Overall Comfort Score: {comfort_metrics['overall_comfort_score']:.3f}")
    
    # Visualize results
    vis = DisparityVisualization()
    
    # Create visualizations
    disp_colored = vis.create_disparity_colormap(disparity)
    gt_colored = vis.create_disparity_colormap(ground_truth.disparity_map)
    error_vis = vis.create_error_visualization(disparity, ground_truth.disparity_map)
    
    # Display results
    cv2.imshow('Left Image', cv2.resize(left, (640, 480)))
    cv2.imshow('Right Image', cv2.resize(right, (640, 480)))
    cv2.imshow('Estimated Disparity', cv2.resize(disp_colored, (640, 480)))
    cv2.imshow('Ground Truth Disparity', cv2.resize(gt_colored, (640, 480)))
    cv2.imshow('Error Visualization', cv2.resize(error_vis, (640, 480)))
    
    print("\nPress any key to close visualization windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'bad_pixel_error': bad_pixel_error,
        'mean_absolute_error': mae,
        'root_mean_square_error': rmse,
        'edge_preservation': edge_score,
        'structural_similarity': ssim,
        **comfort_metrics
    }


def run_comprehensive_tests():
    """Run comprehensive test suite and generate report."""
    print("Running comprehensive test suite...")
    
    # Initialize components
    camera_params = VRCameraParameters()
    generator = AdvancedStereoGenerator(camera_params)
    algorithm = ExampleDisparityAlgorithm()
    evaluator = DisparityEvaluator()
    comfort_analyzer = VRComfortAnalyzer(camera_params)
    
    # Test scenarios
    test_scenarios = [
        ("indoor_normal", {"scene_type": "indoor", "lighting": "normal"}),
        ("indoor_low_light", {"scene_type": "indoor", "lighting": "low"}),
        ("indoor_high_contrast", {"scene_type": "indoor", "lighting": "high_contrast"}),
        ("outdoor_normal", {"scene_type": "outdoor", "lighting": "normal"}),
        ("mixed_scene", {"scene_type": "mixed", "lighting": "normal"}),
    ]
    
    all_results = {}
    
    for test_name, params in test_scenarios:
        print(f"\nTesting scenario: {test_name}")
        
        # Generate test scene
        left, right, ground_truth = generator.create_realistic_scene(**params)
        
        # Compute disparity
        disparity = algorithm(left, right)
        
        # Evaluate
        results = {
            'bad_pixel_error': evaluator.bad_pixel_error(
                disparity, ground_truth.disparity_map
            ),
            'mean_absolute_error': evaluator.mean_absolute_error(
                disparity, ground_truth.disparity_map
            ),
            'edge_preservation': evaluator.edge_preservation_score(
                disparity, ground_truth.disparity_map
            ),
            'structural_similarity': evaluator.structural_similarity(
                disparity, ground_truth.disparity_map
            ),
        }
        
        # Add comfort metrics
        comfort = comfort_analyzer.analyze_comfort(disparity)
        results.update(comfort)
        
        all_results[test_name] = results
        
        print(f"  Bad Pixel Error: {results['bad_pixel_error']:.2f}%")
        print(f"  Comfort Score: {results['overall_comfort_score']:.3f}")
    
    # Generate report
    output_dir = "test_results"
    generate_test_report(all_results, output_dir, "ExampleDisparityAlgorithm")
    
    print(f"\nTest report generated in: {output_dir}/")
    
    return all_results


def run_pytest_suite():
    """Run the full pytest suite."""
    print("Running pytest suite...")
    
    # Create a temporary module to inject our algorithm
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
import numpy as np
import cv2

class ExampleDisparityAlgorithm:
    def __init__(self):
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    
    def __call__(self, left_image, right_image):
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image
        
        disparity = self.stereo.compute(left_gray, right_gray)
        disparity = disparity.astype(np.float32) / 16.0
        disparity[disparity < 0] = 0
        if disparity.max() > 0:
            disparity = (disparity / disparity.max() * 255).astype(np.uint8)
        else:
            disparity = disparity.astype(np.uint8)
        
        return disparity

# Override the fixture
import pytest

@pytest.fixture
def disparity_algorithm():
    return ExampleDisparityAlgorithm()
""")
        temp_file = f.name
    
    # Run pytest with our test file
    pytest_args = [
        'test_vr_disparity_detection.py',
        '-v',  # Verbose
        '-s',  # Show print statements
        '--tb=short',  # Short traceback
        '-k', 'not test_processing_speed and not test_memory_usage'  # Skip performance tests
    ]
    
    # Save current sys.argv and restore after
    old_argv = sys.argv
    sys.argv = ['pytest'] + pytest_args
    
    try:
        # Import and configure pytest
        pytest.main(pytest_args)
    finally:
        sys.argv = old_argv
        # Clean up temp file
        os.unlink(temp_file)


def main():
    """Main function to run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Example usage of VR disparity detection test suite"
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'comprehensive', 'pytest'],
        default='single',
        help='Test mode to run'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_test()
    elif args.mode == 'comprehensive':
        run_comprehensive_tests()
    elif args.mode == 'pytest':
        run_pytest_suite()


if __name__ == "__main__":
    main()