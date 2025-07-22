# VR Video Disparity Detection Test Suite

This comprehensive test suite evaluates algorithms that detect disparity between left and right eye views in VR video frames. The tests are designed based on extensive research into VR video quality assessment, stereo vision benchmarks, and viewer comfort metrics.

## Overview

The test suite evaluates VR disparity detection algorithms across multiple dimensions:

1. **Basic Functionality**: Correct input/output handling and data formats
2. **Accuracy Metrics**: Bad pixel percentage, MAE, RMSE, structural similarity
3. **VR-Specific Challenges**: Wide baselines, lens distortion, temporal consistency
4. **Viewer Comfort**: Vergence-accommodation conflict, excessive disparity detection
5. **Robustness**: Handling of textureless regions, occlusions, specular surfaces

## Requirements

```bash
pip install pytest numpy opencv-python matplotlib scikit-learn
```

## Algorithm Interface

Your disparity detection algorithm must implement the following interface:

```python
def disparity_algorithm(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    """
    Detect disparity between left and right eye images.
    
    Args:
        left_image: Left eye image, shape (H, W, 3), dtype uint8
        right_image: Right eye image, shape (H, W, 3), dtype uint8
    
    Returns:
        Disparity map, shape (H, W), dtype uint8, values in range [0, 255]
        Higher values indicate larger disparity (closer objects)
        Pixels with low confidence or invalid disparity should be set to 0
    """
    # Your implementation here
    pass
```

## Test Categories

### 1. Basic Functionality Tests
- Input/output format validation
- Resolution handling (QVGA to 4K)
- Grayscale and color image support
- Data type and value range checks

### 2. Accuracy Evaluation
- **Bad Pixel Error**: Percentage of pixels with error > threshold (standard metric)
- **Mean Absolute Error (MAE)**: Average pixel-wise error
- **Root Mean Square Error (RMSE)**: Emphasizes larger errors
- **Structural Similarity (SSIM)**: Perceptual quality metric
- **Edge Preservation**: How well depth discontinuities are preserved

### 3. Synthetic Test Scenes
- **Planar Surfaces**: Multiple fronto-parallel planes at different depths
- **Textured Planes**: Random noise, checkerboard, gradient patterns
- **Slanted Surfaces**: Linearly varying disparity
- **Occlusion Scenarios**: Foreground objects occluding background
- **Textureless Regions**: Large uniform areas (challenging for matching)

### 4. VR-Specific Tests
- **Wide Baseline**: Larger than typical 65mm IPD
- **Lens Distortion**: Simulated barrel distortion
- **Vertical Disparity**: Misalignment between views
- **Temporal Consistency**: Stability across video frames
- **Vergence-Accommodation Conflict**: Comfort analysis

### 5. Robustness Tests
- **High Disparity Range**: Very near/far objects
- **Repetitive Patterns**: Aliasing challenges
- **Low Contrast**: Reduced texture visibility
- **Specular Surfaces**: View-dependent reflections
- **Transparent Objects**: See-through materials

## Usage

### Basic Example

```python
from test_vr_disparity_detection import DisparityEvaluator
from test_vr_disparity_utils import SyntheticStereoGenerator

# Your algorithm
def my_disparity_algorithm(left, right):
    # Implementation
    return disparity_map

# Generate test data
generator = SyntheticStereoGenerator()
test_case = generator.create_textured_plane(texture_type="checkerboard")

# Run algorithm
result = my_disparity_algorithm(test_case.left_image, test_case.right_image)

# Evaluate
evaluator = DisparityEvaluator()
bad_pixels = evaluator.bad_pixel_error(result, test_case.ground_truth_disparity)
print(f"Bad pixel error: {bad_pixels:.2f}%")
```

### Running Full Test Suite

```python
# Run with pytest
pytest test_vr_disparity_detection.py -v

# Or use the example script
python example_test_usage.py --mode comprehensive
```

### Custom Test Integration

```python
import pytest

@pytest.fixture
def disparity_algorithm():
    """Provide your algorithm to the test suite."""
    from my_module import MyDisparityAlgorithm
    return MyDisparityAlgorithm()

# Run tests
pytest test_vr_disparity_detection.py -v
```

## Evaluation Metrics

### Bad Pixel Percentage (Most Common)
- Pixels with error > threshold (typically 3 pixels)
- Standard metric in stereo benchmarks
- Lower is better (< 10% is good, < 5% is excellent)

### VR Comfort Metrics
- **Vergence Angle**: Should be < 1.5° for comfort
- **Disparity Gradient**: Rapid changes cause eye strain
- **Screen Disparity**: Physical separation on display

### Temporal Consistency
- Frame-to-frame disparity changes
- Flicker detection
- Important for video applications

## Advanced Features

### Realistic Scene Generation
```python
from test_vr_disparity_utils import AdvancedStereoGenerator, VRCameraParameters

camera = VRCameraParameters(
    baseline=65.0,  # mm
    focal_length=500.0,  # pixels
    resolution=(1920, 1080)
)

generator = AdvancedStereoGenerator(camera)
left, right, ground_truth = generator.create_realistic_scene(
    scene_type="indoor",
    lighting="low",
    add_noise=True,
    add_compression_artifacts=True
)
```

### Visualization Tools
```python
from test_vr_disparity_utils import DisparityVisualization

vis = DisparityVisualization()

# Colormap visualization
colored = vis.create_disparity_colormap(disparity_map)

# Error visualization (green=correct, red=error)
error_vis = vis.create_error_visualization(estimated, ground_truth)

# Confidence overlay
conf_vis = vis.create_confidence_visualization(confidence_map, overlay_on=left_image)
```

### Report Generation
```python
from test_vr_disparity_utils import generate_test_report

results = {
    'test1': {'bad_pixel_error': 5.2, 'mae': 2.1},
    'test2': {'bad_pixel_error': 8.7, 'mae': 3.5}
}

generate_test_report(results, output_dir="reports", algorithm_name="MyAlgorithm")
```

## Interpreting Results

### Good Performance Indicators
- Bad pixel error < 10% on textured regions
- Temporal consistency score > 0.8
- Edge preservation score > 0.7
- Comfort score > 0.7

### Common Issues
- High error in textureless regions (expected)
- Poor temporal consistency (flicker in video)
- Low comfort scores (excessive disparity)
- Poor edge preservation (blurred depth boundaries)

## Research Background

This test suite is based on established evaluation methodologies:

1. **Middlebury Stereo Benchmark**: Standard evaluation metrics and test scenes
2. **Bad Pixel Percentage**: Most widely used accuracy metric
3. **VR Comfort Research**: Vergence-accommodation conflict studies
4. **Perceptual Quality**: SSIM and gradient-based metrics

## Limitations

- Synthetic test data may not capture all real-world complexities
- Ground truth is perfect (real data has measurement errors)
- Comfort metrics are approximations of human perception
- Does not test all possible VR display technologies

## Contributing

To add new test cases or metrics:

1. Add test data generators to `SyntheticStereoGenerator`
2. Add evaluation metrics to `DisparityEvaluator`
3. Add test functions to appropriate test classes
4. Update documentation

## References

- Scharstein & Szeliski (2002): "A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms"
- Middlebury Stereo Vision Benchmark: vision.middlebury.edu/stereo
- ITU-R BT.500: "Methodology for the subjective assessment of the quality of television pictures"
- Hoffman et al. (2008): "Vergence-accommodation conflicts hinder visual performance and cause visual fatigue"