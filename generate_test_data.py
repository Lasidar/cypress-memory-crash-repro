"""
Utility script to generate synthetic stereo pairs for testing channel mismatch detection.
This creates simple test cases with known ground truth.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def create_synthetic_stereo_pair(
    width: int = 1920,
    height: int = 1080,
    baseline: float = 65.0,  # mm, typical human IPD
    focal_length: float = 35.0,  # mm
    num_objects: int = 5,
    add_noise: bool = False,
    swap_views: bool = False
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a synthetic stereo pair with known properties.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        baseline: Stereo baseline in mm (interpupillary distance)
        focal_length: Camera focal length in mm
        num_objects: Number of random objects to add
        add_noise: Whether to add noise to images
        swap_views: Whether to swap left and right views
        
    Returns:
        left_image: Left eye view
        right_image: Right eye view
        metadata: Dictionary with ground truth information
    """
    # Initialize blank images
    left = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background
    right = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Camera parameters (simplified pinhole model)
    cx = width / 2
    cy = height / 2
    fx = fy = focal_length * width / 35.0  # Normalize to sensor size
    
    # Store object information
    objects = []
    
    # Generate random objects at different depths
    np.random.seed(42)  # For reproducibility
    for i in range(num_objects):
        # Random object properties
        obj_x = np.random.randint(width // 4, 3 * width // 4)
        obj_y = np.random.randint(height // 4, 3 * height // 4)
        obj_z = np.random.uniform(500, 3000)  # Depth in mm
        obj_size = np.random.randint(30, 100)
        obj_color = np.random.randint(0, 255, 3).tolist()
        
        # Calculate disparity (in pixels)
        disparity = (baseline * fx) / obj_z
        
        # Draw object in left view
        cv2.rectangle(
            left,
            (obj_x - obj_size // 2, obj_y - obj_size // 2),
            (obj_x + obj_size // 2, obj_y + obj_size // 2),
            obj_color,
            -1
        )
        
        # Draw object in right view with horizontal disparity
        right_x = int(obj_x - disparity)
        if 0 <= right_x - obj_size // 2 and right_x + obj_size // 2 < width:
            cv2.rectangle(
                right,
                (right_x - obj_size // 2, obj_y - obj_size // 2),
                (right_x + obj_size // 2, obj_y + obj_size // 2),
                obj_color,
                -1
            )
            
            # Add occlusion region (visible in left but not right)
            if right_x + obj_size // 2 < obj_x - obj_size // 2:
                cv2.rectangle(
                    left,
                    (right_x + obj_size // 2, obj_y - obj_size // 2),
                    (obj_x - obj_size // 2, obj_y + obj_size // 2),
                    [64, 64, 64],  # Dark gray for occlusion
                    -1
                )
        
        objects.append({
            'position': (obj_x, obj_y, obj_z),
            'size': obj_size,
            'color': obj_color,
            'disparity': disparity
        })
    
    # Add texture patterns for better feature matching
    for i in range(20):
        pt1 = (np.random.randint(0, width), np.random.randint(0, height))
        pt2 = (np.random.randint(0, width), np.random.randint(0, height))
        color = np.random.randint(0, 255, 3).tolist()
        thickness = np.random.randint(1, 3)
        
        cv2.line(left, pt1, pt2, color, thickness)
        # Apply same texture with disparity
        avg_depth = 1500  # mm
        texture_disparity = (baseline * fx) / avg_depth
        pt1_right = (int(pt1[0] - texture_disparity), pt1[1])
        pt2_right = (int(pt2[0] - texture_disparity), pt2[1])
        cv2.line(right, pt1_right, pt2_right, color, thickness)
    
    # Add noise if requested
    if add_noise:
        noise_left = np.random.normal(0, 10, left.shape).astype(np.uint8)
        noise_right = np.random.normal(0, 10, right.shape).astype(np.uint8)
        left = cv2.add(left, noise_left)
        right = cv2.add(right, noise_right)
    
    # Swap views if requested (to simulate channel mismatch)
    if swap_views:
        left, right = right, left
    
    # Create metadata
    metadata = {
        'swapped': swap_views,
        'baseline_mm': baseline,
        'focal_length_mm': focal_length,
        'image_size': (width, height),
        'objects': objects,
        'has_noise': add_noise,
        'expected_mismatch_score': 0.9 if swap_views else 0.1
    }
    
    return left, right, metadata


def create_challenging_cases() -> list:
    """
    Create a set of challenging test cases for robustness testing.
    
    Returns:
        List of (left, right, metadata) tuples
    """
    test_cases = []
    
    # Case 1: Low texture scene
    left = np.ones((480, 640, 3), dtype=np.uint8) * 100
    right = np.ones((480, 640, 3), dtype=np.uint8) * 100
    # Add minimal features
    cv2.circle(left, (320, 240), 50, (255, 255, 255), -1)
    cv2.circle(right, (310, 240), 50, (255, 255, 255), -1)
    test_cases.append((left, right, {'name': 'low_texture', 'swapped': False}))
    
    # Case 2: High noise
    left, right, meta = create_synthetic_stereo_pair(640, 480, add_noise=True)
    meta['name'] = 'high_noise'
    test_cases.append((left, right, meta))
    
    # Case 3: Extreme disparity
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(left, (100, 200), (200, 300), (255, 0, 0), -1)
    cv2.rectangle(right, (400, 200), (500, 300), (255, 0, 0), -1)  # 300px disparity!
    test_cases.append((left, right, {'name': 'extreme_disparity', 'swapped': False}))
    
    # Case 4: Repetitive pattern
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    for x in range(0, 640, 40):
        cv2.line(left, (x, 0), (x, 480), (255, 255, 255), 2)
        cv2.line(right, (x-10, 0), (x-10, 480), (255, 255, 255), 2)
    test_cases.append((left, right, {'name': 'repetitive_pattern', 'swapped': False}))
    
    # Case 5: Swapped views with occlusion violations
    left, right, meta = create_synthetic_stereo_pair(640, 480, swap_views=True)
    meta['name'] = 'swapped_with_occlusions'
    test_cases.append((left, right, meta))
    
    return test_cases


def save_test_images(output_dir: str = "test_data"):
    """Save test images for manual inspection and algorithm testing."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate normal stereo pair
    left, right, meta = create_synthetic_stereo_pair()
    cv2.imwrite(f"{output_dir}/normal_left.png", left)
    cv2.imwrite(f"{output_dir}/normal_right.png", right)
    
    # Generate swapped stereo pair
    left_swap, right_swap, meta_swap = create_synthetic_stereo_pair(swap_views=True)
    cv2.imwrite(f"{output_dir}/swapped_left.png", left_swap)
    cv2.imwrite(f"{output_dir}/swapped_right.png", right_swap)
    
    # Generate challenging cases
    for i, (left, right, meta) in enumerate(create_challenging_cases()):
        name = meta.get('name', f'case_{i}')
        cv2.imwrite(f"{output_dir}/{name}_left.png", left)
        cv2.imwrite(f"{output_dir}/{name}_right.png", right)
    
    print(f"Test images saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    left, right, metadata = create_synthetic_stereo_pair()
    print(f"Created stereo pair: {metadata['image_size']}")
    print(f"Swapped: {metadata['swapped']}")
    print(f"Number of objects: {len(metadata['objects'])}")
    
    # Save test images
    save_test_images()