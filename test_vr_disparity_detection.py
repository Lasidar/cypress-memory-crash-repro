"""
Comprehensive pytest suite for evaluating VR video disparity detection algorithms.

This test suite evaluates algorithms that detect disparity between left and right eye views
in VR video frames. The algorithm under test should accept two numpy arrays (left and right
eye images) and return a disparity map as a numpy array of shape (height, width) with dtype
uint8, where values are normalized to 0-255 range.
"""

import pytest
import numpy as np
from typing import Dict, Tuple, Callable, Optional
import cv2
from dataclasses import dataclass
from enum import Enum
import warnings


class DisparityEvaluationMetric(Enum):
    """Standard metrics for disparity map evaluation."""
    BAD_PIXEL_ERROR = "bad_pixel_error"  # Percentage of pixels with error > threshold
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    ROOT_MEAN_SQUARE_ERROR = "root_mean_square_error"
    ENDPOINT_ERROR = "endpoint_error"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    EDGE_PRESERVATION = "edge_preservation"
    SMOOTHNESS = "smoothness"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


@dataclass
class DisparityTestCase:
    """Container for a disparity test case."""
    name: str
    left_image: np.ndarray
    right_image: np.ndarray
    ground_truth_disparity: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class SyntheticStereoGenerator:
    """Generate synthetic stereo pairs with known ground truth disparity."""
    
    @staticmethod
    def create_planar_scene(
        width: int = 640,
        height: int = 480,
        baseline: float = 65.0,  # mm, typical IPD
        focal_length: float = 500.0,  # pixels
        num_planes: int = 3
    ) -> DisparityTestCase:
        """Create a scene with multiple fronto-parallel planes at different depths."""
        left_image = np.zeros((height, width, 3), dtype=np.uint8)
        right_image = np.zeros((height, width, 3), dtype=np.uint8)
        disparity_map = np.zeros((height, width), dtype=np.float32)
        
        # Create planes at different depths
        depths = np.linspace(500, 2000, num_planes)  # mm
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, depth in enumerate(depths):
            # Calculate disparity for this depth
            disparity = (baseline * focal_length) / depth
            
            # Define region for this plane
            y_start = int(i * height / num_planes)
            y_end = int((i + 1) * height / num_planes)
            
            # Fill regions
            color = colors[i % len(colors)]
            left_image[y_start:y_end, :] = color
            
            # Shift right image by disparity
            shift = int(disparity)
            if shift > 0:
                right_image[y_start:y_end, shift:] = left_image[y_start:y_end, :-shift]
            else:
                right_image[y_start:y_end, :] = left_image[y_start:y_end, :]
            
            disparity_map[y_start:y_end, :] = disparity
        
        # Normalize disparity to 0-255 range
        disparity_normalized = ((disparity_map / disparity_map.max()) * 255).astype(np.uint8)
        
        return DisparityTestCase(
            name="planar_scene",
            left_image=left_image,
            right_image=right_image,
            ground_truth_disparity=disparity_normalized,
            metadata={"baseline": baseline, "focal_length": focal_length, "depths": depths}
        )
    
    @staticmethod
    def create_textured_plane(
        width: int = 640,
        height: int = 480,
        disparity: int = 20,
        texture_type: str = "random"
    ) -> DisparityTestCase:
        """Create a textured plane with known disparity."""
        if texture_type == "random":
            # Random noise texture
            left_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        elif texture_type == "checkerboard":
            # Checkerboard pattern
            block_size = 20
            left_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if ((i // block_size) + (j // block_size)) % 2 == 0:
                        left_image[i:i+block_size, j:j+block_size] = 255
        elif texture_type == "gradient":
            # Gradient texture
            x_grad = np.linspace(0, 255, width)
            y_grad = np.linspace(0, 255, height)
            xx, yy = np.meshgrid(x_grad, y_grad)
            left_image = np.stack([xx, yy, (xx + yy) / 2], axis=2).astype(np.uint8)
        else:
            raise ValueError(f"Unknown texture type: {texture_type}")
        
        # Create right image by shifting
        right_image = np.zeros_like(left_image)
        if disparity > 0:
            right_image[:, disparity:] = left_image[:, :-disparity]
        else:
            right_image = left_image.copy()
        
        # Create ground truth disparity map
        disparity_map = np.full((height, width), disparity, dtype=np.uint8)
        
        return DisparityTestCase(
            name=f"textured_plane_{texture_type}",
            left_image=left_image,
            right_image=right_image,
            ground_truth_disparity=disparity_map,
            metadata={"texture_type": texture_type, "disparity": disparity}
        )
    
    @staticmethod
    def create_slanted_plane(
        width: int = 640,
        height: int = 480,
        min_disparity: int = 10,
        max_disparity: int = 50
    ) -> DisparityTestCase:
        """Create a slanted plane with linearly varying disparity."""
        # Create textured surface
        left_image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        
        # Add some structure
        for i in range(0, height, 50):
            left_image[i:i+5, :] = 255
        for j in range(0, width, 50):
            left_image[:, j:j+5] = 255
        
        # Create disparity gradient
        disparity_map = np.zeros((height, width), dtype=np.float32)
        for x in range(width):
            disparity = min_disparity + (max_disparity - min_disparity) * (x / width)
            disparity_map[:, x] = disparity
        
        # Create right image with varying shifts
        right_image = np.zeros_like(left_image)
        for x in range(width):
            shift = int(disparity_map[0, x])
            if x + shift < width:
                right_image[:, x + shift] = left_image[:, x]
        
        # Normalize disparity
        disparity_normalized = ((disparity_map / disparity_map.max()) * 255).astype(np.uint8)
        
        return DisparityTestCase(
            name="slanted_plane",
            left_image=left_image,
            right_image=right_image,
            ground_truth_disparity=disparity_normalized,
            metadata={"min_disparity": min_disparity, "max_disparity": max_disparity}
        )
    
    @staticmethod
    def create_occlusion_test(
        width: int = 640,
        height: int = 480,
        foreground_disparity: int = 40,
        background_disparity: int = 10
    ) -> DisparityTestCase:
        """Create a scene with occlusions to test algorithm robustness."""
        left_image = np.full((height, width, 3), 100, dtype=np.uint8)  # Gray background
        right_image = np.full((height, width, 3), 100, dtype=np.uint8)
        disparity_map = np.full((height, width), background_disparity, dtype=np.uint8)
        
        # Add foreground object
        obj_width, obj_height = 200, 200
        obj_x, obj_y = width // 2 - obj_width // 2, height // 2 - obj_height // 2
        
        # Draw object in left image
        left_image[obj_y:obj_y+obj_height, obj_x:obj_x+obj_width] = [255, 0, 0]  # Red
        
        # Draw object in right image with disparity shift
        right_obj_x = obj_x - foreground_disparity
        if right_obj_x >= 0:
            right_image[obj_y:obj_y+obj_height, right_obj_x:right_obj_x+obj_width] = [255, 0, 0]
        
        # Update disparity map
        disparity_map[obj_y:obj_y+obj_height, obj_x:obj_x+obj_width] = foreground_disparity
        
        # Normalize disparity
        disparity_normalized = ((disparity_map / disparity_map.max()) * 255).astype(np.uint8)
        
        return DisparityTestCase(
            name="occlusion_test",
            left_image=left_image,
            right_image=right_image,
            ground_truth_disparity=disparity_normalized,
            metadata={
                "foreground_disparity": foreground_disparity,
                "background_disparity": background_disparity,
                "has_occlusions": True
            }
        )
    
    @staticmethod
    def create_textureless_region(
        width: int = 640,
        height: int = 480,
        disparity: int = 25
    ) -> DisparityTestCase:
        """Create a scene with large textureless regions to test algorithm robustness."""
        left_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some textured regions
        # Top region - textured
        left_image[:height//3, :] = np.random.randint(0, 256, (height//3, width, 3), dtype=np.uint8)
        
        # Middle region - textureless
        left_image[height//3:2*height//3, :] = 128  # Uniform gray
        
        # Bottom region - textured
        left_image[2*height//3:, :] = np.random.randint(0, 256, (height//3, width, 3), dtype=np.uint8)
        
        # Create right image with shift
        right_image = np.zeros_like(left_image)
        if disparity > 0:
            right_image[:, disparity:] = left_image[:, :-disparity]
        
        # Ground truth disparity
        disparity_map = np.full((height, width), disparity, dtype=np.uint8)
        
        return DisparityTestCase(
            name="textureless_region",
            left_image=left_image,
            right_image=right_image,
            ground_truth_disparity=disparity_map,
            metadata={"has_textureless": True, "disparity": disparity}
        )


class DisparityEvaluator:
    """Evaluate disparity maps using various metrics."""
    
    @staticmethod
    def bad_pixel_error(
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 3.0,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate percentage of bad pixels (error > threshold).
        This is the most common metric in stereo evaluation.
        """
        if mask is None:
            mask = np.ones_like(ground_truth, dtype=bool)
        
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return 100.0
        
        error = np.abs(estimated.astype(float) - ground_truth.astype(float))
        bad_pixels = (error > threshold) & mask
        
        return (bad_pixels.sum() / valid_pixels) * 100.0
    
    @staticmethod
    def mean_absolute_error(
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Calculate mean absolute error."""
        if mask is None:
            mask = np.ones_like(ground_truth, dtype=bool)
        
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return float('inf')
        
        error = np.abs(estimated.astype(float) - ground_truth.astype(float))
        return (error * mask).sum() / valid_pixels
    
    @staticmethod
    def root_mean_square_error(
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Calculate root mean square error."""
        if mask is None:
            mask = np.ones_like(ground_truth, dtype=bool)
        
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return float('inf')
        
        error = (estimated.astype(float) - ground_truth.astype(float)) ** 2
        return np.sqrt((error * mask).sum() / valid_pixels)
    
    @staticmethod
    def edge_preservation_score(
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 10.0
    ) -> float:
        """
        Evaluate how well edges are preserved in the disparity map.
        Returns a score between 0 and 1, where 1 is perfect preservation.
        """
        # Compute gradients
        gt_grad_x = cv2.Sobel(ground_truth, cv2.CV_64F, 1, 0, ksize=3)
        gt_grad_y = cv2.Sobel(ground_truth, cv2.CV_64F, 0, 1, ksize=3)
        gt_grad_mag = np.sqrt(gt_grad_x**2 + gt_grad_y**2)
        
        est_grad_x = cv2.Sobel(estimated, cv2.CV_64F, 1, 0, ksize=3)
        est_grad_y = cv2.Sobel(estimated, cv2.CV_64F, 0, 1, ksize=3)
        est_grad_mag = np.sqrt(est_grad_x**2 + est_grad_y**2)
        
        # Find edge pixels
        edge_mask = gt_grad_mag > threshold
        
        if edge_mask.sum() == 0:
            return 1.0
        
        # Compare gradient magnitudes at edge locations
        diff = np.abs(gt_grad_mag - est_grad_mag)
        score = 1.0 - (diff[edge_mask].mean() / (gt_grad_mag[edge_mask].mean() + 1e-6))
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def smoothness_score(disparity_map: np.ndarray) -> float:
        """
        Evaluate smoothness of disparity map.
        Lower values indicate smoother maps.
        """
        # Compute gradients
        grad_x = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Total variation
        tv = np.abs(grad_x).mean() + np.abs(grad_y).mean()
        
        return tv
    
    @staticmethod
    def structural_similarity(
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        window_size: int = 11
    ) -> float:
        """
        Calculate structural similarity index (SSIM) for disparity maps.
        Returns a value between -1 and 1, where 1 indicates perfect similarity.
        """
        # Constants for stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Convert to float
        est = estimated.astype(np.float64)
        gt = ground_truth.astype(np.float64)
        
        # Create Gaussian kernel
        kernel = cv2.getGaussianKernel(window_size, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        # Calculate local means
        mu1 = cv2.filter2D(gt, -1, window)
        mu2 = cv2.filter2D(est, -1, window)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate local variances and covariance
        sigma1_sq = cv2.filter2D(gt ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(est ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(gt * est, -1, window) - mu1_mu2
        
        # SSIM calculation
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


# Test fixtures
@pytest.fixture
def synthetic_test_cases():
    """Generate a set of synthetic test cases."""
    generator = SyntheticStereoGenerator()
    
    test_cases = [
        generator.create_planar_scene(),
        generator.create_textured_plane(texture_type="random"),
        generator.create_textured_plane(texture_type="checkerboard"),
        generator.create_textured_plane(texture_type="gradient"),
        generator.create_slanted_plane(),
        generator.create_occlusion_test(),
        generator.create_textureless_region()
    ]
    
    return test_cases


@pytest.fixture
def disparity_algorithm():
    """
    Fixture for the disparity detection algorithm.
    This should be replaced with the actual algorithm being tested.
    """
    def dummy_algorithm(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """Dummy algorithm that returns a random disparity map."""
        height, width = left_image.shape[:2]
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    return dummy_algorithm


# Test classes
class TestDisparityAlgorithmBasicFunctionality:
    """Test basic functionality and input/output requirements."""
    
    def test_algorithm_accepts_numpy_arrays(self, disparity_algorithm):
        """Test that algorithm accepts numpy arrays as input."""
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = disparity_algorithm(left, right)
        assert isinstance(result, np.ndarray), "Algorithm should return numpy array"
    
    def test_output_shape_matches_input(self, disparity_algorithm):
        """Test that output disparity map has correct shape."""
        height, width = 480, 640
        left = np.zeros((height, width, 3), dtype=np.uint8)
        right = np.zeros((height, width, 3), dtype=np.uint8)
        
        result = disparity_algorithm(left, right)
        assert result.shape == (height, width), f"Expected shape {(height, width)}, got {result.shape}"
    
    def test_output_dtype_is_uint8(self, disparity_algorithm):
        """Test that output has correct data type."""
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = disparity_algorithm(left, right)
        assert result.dtype == np.uint8, f"Expected dtype uint8, got {result.dtype}"
    
    def test_output_value_range(self, disparity_algorithm):
        """Test that output values are in valid range [0, 255]."""
        left = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        result = disparity_algorithm(left, right)
        assert result.min() >= 0, f"Minimum value {result.min()} is less than 0"
        assert result.max() <= 255, f"Maximum value {result.max()} is greater than 255"
    
    def test_handles_grayscale_input(self, disparity_algorithm):
        """Test that algorithm can handle grayscale input."""
        left = np.zeros((480, 640), dtype=np.uint8)
        right = np.zeros((480, 640), dtype=np.uint8)
        
        # Convert to 3-channel for consistency
        left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)
        
        result = disparity_algorithm(left_rgb, right_rgb)
        assert result.shape == (480, 640)
    
    def test_handles_different_resolutions(self, disparity_algorithm):
        """Test algorithm with different image resolutions."""
        resolutions = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        
        for width, height in resolutions:
            left = np.zeros((height, width, 3), dtype=np.uint8)
            right = np.zeros((height, width, 3), dtype=np.uint8)
            
            result = disparity_algorithm(left, right)
            assert result.shape == (height, width), \
                f"Failed for resolution {width}x{height}"


class TestDisparityAccuracy:
    """Test accuracy of disparity estimation."""
    
    def test_planar_surfaces(self, disparity_algorithm, synthetic_test_cases):
        """Test accuracy on planar surfaces."""
        evaluator = DisparityEvaluator()
        
        for test_case in synthetic_test_cases:
            if "planar" in test_case.name:
                result = disparity_algorithm(test_case.left_image, test_case.right_image)
                
                # Calculate metrics
                bad_pixel_pct = evaluator.bad_pixel_error(
                    result, test_case.ground_truth_disparity, threshold=3.0
                )
                mae = evaluator.mean_absolute_error(result, test_case.ground_truth_disparity)
                
                # Log results (not asserting specific thresholds as algorithm is unknown)
                print(f"\n{test_case.name}:")
                print(f"  Bad pixel %: {bad_pixel_pct:.2f}")
                print(f"  MAE: {mae:.2f}")
    
    def test_textured_regions(self, disparity_algorithm):
        """Test performance on different texture types."""
        generator = SyntheticStereoGenerator()
        evaluator = DisparityEvaluator()
        
        texture_types = ["random", "checkerboard", "gradient"]
        results = {}
        
        for texture_type in texture_types:
            test_case = generator.create_textured_plane(texture_type=texture_type)
            result = disparity_algorithm(test_case.left_image, test_case.right_image)
            
            bad_pixel_pct = evaluator.bad_pixel_error(
                result, test_case.ground_truth_disparity
            )
            results[texture_type] = bad_pixel_pct
        
        # Textured regions should generally perform better than textureless
        assert results["random"] < 100.0, "Algorithm should detect some correct disparities"
    
    def test_occlusion_handling(self, disparity_algorithm):
        """Test algorithm's handling of occlusions."""
        generator = SyntheticStereoGenerator()
        test_case = generator.create_occlusion_test()
        
        result = disparity_algorithm(test_case.left_image, test_case.right_image)
        
        # Check if algorithm produces reasonable disparities in occluded regions
        # (This is a basic sanity check)
        assert result.mean() > 0, "Algorithm should produce non-zero disparities"
    
    def test_textureless_regions(self, disparity_algorithm):
        """Test performance in textureless regions."""
        generator = SyntheticStereoGenerator()
        evaluator = DisparityEvaluator()
        
        test_case = generator.create_textureless_region()
        result = disparity_algorithm(test_case.left_image, test_case.right_image)
        
        # Analyze performance in different regions
        height = result.shape[0]
        
        # Textured top region
        top_region = result[:height//3, :]
        top_gt = test_case.ground_truth_disparity[:height//3, :]
        top_error = evaluator.bad_pixel_error(top_region, top_gt)
        
        # Textureless middle region
        middle_region = result[height//3:2*height//3, :]
        middle_gt = test_case.ground_truth_disparity[height//3:2*height//3, :]
        middle_error = evaluator.bad_pixel_error(middle_region, middle_gt)
        
        print(f"\nTextureless region test:")
        print(f"  Textured region error: {top_error:.2f}%")
        print(f"  Textureless region error: {middle_error:.2f}%")
        
        # Textureless regions typically have higher error
        # This is expected behavior for most algorithms


class TestDisparityQualityMetrics:
    """Test various quality aspects of disparity maps."""
    
    def test_edge_preservation(self, disparity_algorithm):
        """Test how well edges are preserved."""
        generator = SyntheticStereoGenerator()
        evaluator = DisparityEvaluator()
        
        # Create scene with strong edges
        test_case = generator.create_planar_scene(num_planes=5)
        result = disparity_algorithm(test_case.left_image, test_case.right_image)
        
        edge_score = evaluator.edge_preservation_score(
            result, test_case.ground_truth_disparity
        )
        
        print(f"\nEdge preservation score: {edge_score:.3f}")
        assert 0 <= edge_score <= 1, "Edge score should be in [0, 1]"
    
    def test_smoothness(self, disparity_algorithm):
        """Test smoothness of disparity maps."""
        generator = SyntheticStereoGenerator()
        evaluator = DisparityEvaluator()
        
        # Test on smooth surface
        test_case = generator.create_textured_plane(disparity=30)
        result = disparity_algorithm(test_case.left_image, test_case.right_image)
        
        smoothness = evaluator.smoothness_score(result)
        print(f"\nSmoothness score: {smoothness:.3f}")
        
        # Lower values indicate smoother maps
        assert smoothness >= 0, "Smoothness should be non-negative"
    
    def test_structural_similarity(self, disparity_algorithm, synthetic_test_cases):
        """Test structural similarity of estimated disparity maps."""
        evaluator = DisparityEvaluator()
        
        for test_case in synthetic_test_cases:
            result = disparity_algorithm(test_case.left_image, test_case.right_image)
            
            ssim = evaluator.structural_similarity(
                result, test_case.ground_truth_disparity
            )
            
            print(f"\n{test_case.name} SSIM: {ssim:.3f}")
            assert -1 <= ssim <= 1, "SSIM should be in [-1, 1]"


class TestRobustnessAndEdgeCases:
    """Test algorithm robustness to various challenging conditions."""
    
    def test_high_disparity_range(self, disparity_algorithm):
        """Test with large disparity values."""
        generator = SyntheticStereoGenerator()
        
        # Create scene with very large disparities
        test_case = generator.create_slanted_plane(
            min_disparity=5,
            max_disparity=100  # Large disparity range
        )
        
        result = disparity_algorithm(test_case.left_image, test_case.right_image)
        
        # Check that algorithm handles large disparities
        assert result.max() > result.min(), "Algorithm should detect disparity variation"
    
    def test_repetitive_patterns(self, disparity_algorithm):
        """Test with repetitive patterns that can cause aliasing."""
        width, height = 640, 480
        
        # Create repetitive pattern
        left_image = np.zeros((height, width, 3), dtype=np.uint8)
        pattern_width = 20
        for i in range(0, width, pattern_width * 2):
            left_image[:, i:i+pattern_width] = 255
        
        # Create right image with ambiguous shift
        right_image = np.roll(left_image, shift=pattern_width, axis=1)
        
        result = disparity_algorithm(left_image, right_image)
        
        # Algorithm should produce some result even with ambiguous input
        assert result is not None
        assert result.shape == (height, width)
    
    def test_low_texture_contrast(self, disparity_algorithm):
        """Test with low contrast images."""
        generator = SyntheticStereoGenerator()
        
        # Create low contrast scene
        test_case = generator.create_textured_plane(texture_type="random")
        
        # Reduce contrast
        low_contrast_left = (test_case.left_image * 0.2 + 128).astype(np.uint8)
        low_contrast_right = (test_case.right_image * 0.2 + 128).astype(np.uint8)
        
        result = disparity_algorithm(low_contrast_left, low_contrast_right)
        
        # Algorithm should still produce output
        assert result is not None
        assert 0 <= result.min() <= result.max() <= 255
    
    def test_specular_surfaces(self, disparity_algorithm):
        """Test with simulated specular reflections."""
        width, height = 640, 480
        
        # Create base image
        left_image = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
        
        # Add specular highlights
        cv2.circle(left_image, (width//3, height//2), 50, (255, 255, 255), -1)
        cv2.circle(left_image, (2*width//3, height//2), 50, (255, 255, 255), -1)
        
        # Right image with shifted highlights (simulating view-dependent reflections)
        right_image = left_image.copy()
        right_image[:, 20:] = left_image[:, :-20]
        
        # Shift highlights differently to simulate specular behavior
        cv2.circle(right_image, (width//3 - 25, height//2), 50, (255, 255, 255), -1)
        cv2.circle(right_image, (2*width//3 - 25, height//2), 50, (255, 255, 255), -1)
        
        result = disparity_algorithm(left_image, right_image)
        
        # Check that algorithm doesn't crash with specular surfaces
        assert result is not None
    
    def test_transparent_objects(self, disparity_algorithm):
        """Test with simulated transparent objects."""
        width, height = 640, 480
        
        # Background
        background = np.full((height, width, 3), 100, dtype=np.uint8)
        
        # Foreground "transparent" object (blended)
        foreground = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(foreground, (width//4, height//4), 
                     (3*width//4, 3*height//4), (200, 200, 200), -1)
        
        # Blend to simulate transparency
        alpha = 0.5
        left_image = (alpha * foreground + (1 - alpha) * background).astype(np.uint8)
        
        # Right image with disparity
        right_image = np.zeros_like(left_image)
        shift = 30
        right_image[:, shift:] = left_image[:, :-shift]
        
        result = disparity_algorithm(left_image, right_image)
        
        # Algorithm should handle transparent regions
        assert result is not None


class TestVRSpecificChallenges:
    """Test challenges specific to VR video content."""
    
    def test_wide_baseline(self, disparity_algorithm):
        """Test with wide baseline typical in VR capture."""
        generator = SyntheticStereoGenerator()
        
        # Create scene with wide baseline (larger disparities)
        test_case = generator.create_planar_scene(
            baseline=100.0  # Wider than typical 65mm IPD
        )
        
        result = disparity_algorithm(test_case.left_image, test_case.right_image)
        
        # Check that algorithm handles wide baseline
        assert result is not None
        assert result.mean() > 0, "Should detect non-zero disparities"
    
    def test_lens_distortion_effects(self, disparity_algorithm):
        """Test robustness to lens distortion artifacts."""
        generator = SyntheticStereoGenerator()
        test_case = generator.create_textured_plane(texture_type="checkerboard")
        
        # Simulate barrel distortion
        height, width = test_case.left_image.shape[:2]
        cx, cy = width // 2, height // 2
        
        # Create distortion maps
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        
        for y in range(height):
            for x in range(width):
                dx = x - cx
                dy = y - cy
                r = np.sqrt(dx*dx + dy*dy)
                
                # Barrel distortion
                factor = 1 + 0.0001 * r
                map_x[y, x] = cx + dx * factor
                map_y[y, x] = cy + dy * factor
        
        # Apply distortion
        distorted_left = cv2.remap(test_case.left_image, map_x, map_y, cv2.INTER_LINEAR)
        distorted_right = cv2.remap(test_case.right_image, map_x, map_y, cv2.INTER_LINEAR)
        
        result = disparity_algorithm(distorted_left, distorted_right)
        
        # Algorithm should handle mild distortion
        assert result is not None
    
    def test_vertical_disparity(self, disparity_algorithm):
        """Test handling of vertical disparity (misalignment)."""
        generator = SyntheticStereoGenerator()
        test_case = generator.create_textured_plane()
        
        # Add small vertical shift to simulate misalignment
        shifted_right = np.roll(test_case.right_image, shift=5, axis=0)
        
        # Algorithm should handle small vertical disparities
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about vertical disparity
            result = disparity_algorithm(test_case.left_image, shifted_right)
        
        assert result is not None
    
    def test_temporal_consistency(self, disparity_algorithm):
        """Test temporal consistency across frames."""
        generator = SyntheticStereoGenerator()
        
        # Generate sequence of similar frames
        base_case = generator.create_textured_plane(texture_type="gradient")
        
        # Create slight variations to simulate video sequence
        sequence_results = []
        for i in range(5):
            # Add small noise to simulate temporal changes
            noise = np.random.normal(0, 2, base_case.left_image.shape).astype(np.int16)
            
            left_frame = np.clip(base_case.left_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            right_frame = np.clip(base_case.right_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            result = disparity_algorithm(left_frame, right_frame)
            sequence_results.append(result)
        
        # Check temporal consistency
        for i in range(1, len(sequence_results)):
            diff = np.abs(sequence_results[i].astype(float) - sequence_results[i-1].astype(float))
            mean_diff = diff.mean()
            
            print(f"\nTemporal difference frame {i-1} to {i}: {mean_diff:.2f}")
            
            # Large changes between frames indicate temporal inconsistency
            assert mean_diff < 50, "Disparity should be temporally consistent"


class TestPerformanceAndEfficiency:
    """Test computational performance aspects."""
    
    def test_processing_speed(self, disparity_algorithm):
        """Measure processing time for different resolutions."""
        import time
        
        resolutions = [
            (320, 240, "QVGA"),
            (640, 480, "VGA"),
            (1280, 720, "HD"),
            (1920, 1080, "Full HD")
        ]
        
        print("\nProcessing speed test:")
        for width, height, name in resolutions:
            left = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            right = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            start_time = time.time()
            result = disparity_algorithm(left, right)
            elapsed_time = time.time() - start_time
            
            fps = 1.0 / elapsed_time if elapsed_time > 0 else float('inf')
            print(f"  {name} ({width}x{height}): {elapsed_time:.3f}s ({fps:.1f} FPS)")
    
    def test_memory_usage(self, disparity_algorithm):
        """Test that algorithm doesn't have memory leaks."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run algorithm multiple times
        for i in range(10):
            left = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            right = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            
            result = disparity_algorithm(left, right)
            del result
            
            if i % 5 == 0:
                gc.collect()
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory usage: Initial={initial_memory:.1f}MB, "
              f"Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
        
        # Allow some memory increase but flag potential leaks
        assert memory_increase < 100, f"Potential memory leak: {memory_increase:.1f}MB increase"


# Parameterized tests for comprehensive evaluation
@pytest.mark.parametrize("disparity", [10, 20, 30, 40, 50])
def test_known_disparity_accuracy(disparity_algorithm, disparity):
    """Test accuracy for known disparity values."""
    generator = SyntheticStereoGenerator()
    evaluator = DisparityEvaluator()
    
    test_case = generator.create_textured_plane(disparity=disparity)
    result = disparity_algorithm(test_case.left_image, test_case.right_image)
    
    # Scale ground truth to match the expected disparity value
    scaled_gt = np.full_like(test_case.ground_truth_disparity, disparity)
    
    mae = evaluator.mean_absolute_error(result, scaled_gt)
    print(f"\nDisparity {disparity}: MAE = {mae:.2f}")


@pytest.mark.parametrize("texture_level", ["high", "medium", "low", "none"])
def test_texture_dependency(disparity_algorithm, texture_level):
    """Test how texture level affects accuracy."""
    width, height = 640, 480
    disparity = 25
    
    # Create images with different texture levels
    if texture_level == "high":
        left_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif texture_level == "medium":
        left_image = np.random.randint(100, 156, (height, width, 3), dtype=np.uint8)
        # Add some structure
        for i in range(0, height, 50):
            left_image[i:i+2, :] = 255
    elif texture_level == "low":
        left_image = np.full((height, width, 3), 128, dtype=np.uint8)
        # Add minimal texture
        left_image[::100, ::100] = 255
    else:  # none
        left_image = np.full((height, width, 3), 128, dtype=np.uint8)
    
    # Create right image
    right_image = np.zeros_like(left_image)
    right_image[:, disparity:] = left_image[:, :-disparity]
    
    result = disparity_algorithm(left_image, right_image)
    
    print(f"\nTexture level '{texture_level}': "
          f"Mean disparity = {result.mean():.1f}, Std = {result.std():.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])