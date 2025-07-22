# VR Video Color Mismatch Detection Test Suite

This test suite provides comprehensive pytest coverage for evaluating algorithms that detect color mismatches between left and right eye views in VR video frames.

## Overview

Color mismatches in VR/stereoscopic video can occur due to various factors:
- Camera calibration differences
- Lens variations between cameras
- Different lighting conditions captured by each camera
- Object reflections from different viewing angles
- Post-processing inconsistencies
- Beam splitter artifacts in stereoscopic rigs

These mismatches can lead to viewer discomfort, eye strain, and headaches, making their detection crucial for VR video quality control.

## Test Structure

The test suite is organized into three main test classes:

### 1. TestVRColorMismatchDetection
Main test class covering various color mismatch scenarios:
- Perfect match (no mismatch)
- Global brightness differences
- Color channel shifts
- Local color mismatches (glares/reflections)
- Gamma/contrast differences
- Saturation mismatches
- Hue shifts
- Low confidence regions handling
- Extreme color mismatches
- Edge cases and error handling
- Different image sizes
- Grayscale images
- Real-world mixed artifacts
- Temporal consistency for video sequences
- Performance benchmarking

### 2. TestColorMismatchMetrics
Tests for validating individual metrics:
- Color mismatch score range validation [0, 1]
- Valid confidence ratio range validation [0, 1]
- Quality level categories ('good', 'medium', 'poor', 'severe', 'error')
- Color difference metrics consistency
- Metric correlations

### 3. TestVRColorMismatchIntegration
Integration tests for the complete pipeline:
- Synthetic VR frame generation with known mismatches
- Full pipeline testing with various mismatch types

## Expected Algorithm Interface

The test suite assumes the color mismatch detection algorithm accepts two numpy arrays (left and right eye images) and returns a dictionary with the following metrics:

```python
{
    'color_mismatch_score': float,      # Overall mismatch score [0, 1]
    'valid_confidence_ratio': float,    # Ratio of valid correspondence regions [0, 1]
    'quality_level': str,              # Quality assessment: 'good', 'medium', 'poor', 'severe', 'error'
    'mean_color_difference': float,    # Mean color difference in valid regions
    'max_color_difference': float,     # Maximum color difference in valid regions
    'std_color_difference': float      # Standard deviation of color differences
}
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running All Tests
```bash
pytest test_vr_color_mismatch_detection.py -v
```

### Running Specific Test Classes
```bash
# Run only the main detection tests
pytest test_vr_color_mismatch_detection.py::TestVRColorMismatchDetection -v

# Run only the metrics validation tests
pytest test_vr_color_mismatch_detection.py::TestColorMismatchMetrics -v

# Run only the integration tests
pytest test_vr_color_mismatch_detection.py::TestVRColorMismatchIntegration -v
```

### Running Specific Tests
```bash
# Run a specific test
pytest test_vr_color_mismatch_detection.py::TestVRColorMismatchDetection::test_global_brightness_mismatch -v
```

### Running Performance Tests
```bash
# Run tests marked as performance tests
pytest test_vr_color_mismatch_detection.py -v -m performance
```

## Adapting for Your Algorithm

To use these tests with your actual color mismatch detection algorithm:

1. Replace the mock `color_mismatch_detector` fixture with your actual function
2. Ensure your function returns the expected dictionary structure
3. Adjust the expected values in the tests based on your algorithm's behavior
4. Remove the `patch.object` decorators and use your actual function

Example adaptation:

```python
from your_module import detect_color_mismatch

@pytest.fixture
def color_mismatch_detector():
    return detect_color_mismatch

def test_identical_images_no_mismatch(self, sample_stereo_pair, color_mismatch_detector):
    left_image, right_image = sample_stereo_pair
    result = color_mismatch_detector(left_image, right_image)
    
    assert result['color_mismatch_score'] == 0.0
    assert result['quality_level'] == 'good'
    assert result['mean_color_difference'] == 0.0
```

## Test Coverage

The test suite covers:
- **15 main detection scenarios** testing different types of color mismatches
- **5 metric validation tests** ensuring output consistency
- **Integration tests** with synthetic data generation
- **Edge cases** including empty images, size mismatches, and error conditions
- **Performance benchmarking** for real-time VR requirements

## Color Mismatch Types Tested

1. **Global Mismatches**
   - Brightness differences
   - Gamma/contrast variations
   - Overall color shifts

2. **Channel-Specific Mismatches**
   - Individual RGB channel shifts
   - Saturation differences
   - Hue shifts

3. **Local Mismatches**
   - Glares and reflections
   - Regional color variations
   - Spatially-varying artifacts

4. **Temporal Aspects**
   - Frame-to-frame consistency
   - Gradual color drift detection

## Quality Levels

The tests expect the following quality level classifications based on color mismatch severity:
- **good**: Minimal or no color mismatch (score < 0.2)
- **medium**: Noticeable but acceptable mismatch (0.2 ≤ score < 0.5)
- **poor**: Significant mismatch affecting viewing comfort (0.5 ≤ score < 0.8)
- **severe**: Extreme mismatch likely causing discomfort (score ≥ 0.8)
- **error**: Processing error or invalid input

## References

This test suite is based on research in VR video quality assessment and color mismatch detection, particularly focusing on:
- Stereoscopic 3D video artifacts and their impact on viewer comfort
- Color transfer and correction methods for stereoscopic content
- Real-world datasets of VR video with color mismatches
- Objective quality metrics for stereoscopic video assessment

