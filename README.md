# Channel Mismatch Detection Test Suite for VR Video

This test suite provides comprehensive evaluation for algorithms that detect channel mismatch (swapped left/right eye views) in VR and stereoscopic video frames.

## Overview

Channel mismatch is a critical artifact in stereoscopic/VR video where the left and right eye views are incorrectly assigned or swapped. This causes severe viewer discomfort, headaches, and complete loss of proper depth perception. The test suite evaluates detection algorithms based on multiple criteria derived from stereoscopic vision principles.

## Background

Based on research from leading institutions in stereoscopic video quality assessment, channel mismatch detection algorithms typically analyze:

1. **Disparity patterns** - Objects should have consistent horizontal disparity between views
2. **Occlusion constraints** - Left-view occlusions should appear on the right side of objects
3. **Epipolar geometry** - Feature correspondences must satisfy epipolar constraints
4. **Depth map validity** - Computed depth maps should be smooth and consistent
5. **Geometric constraints** - Minimal vertical disparity in properly rectified stereo

## Test Categories

### 1. Basic Functionality Tests (`TestBasicFunctionality`)
- Verifies API compliance and return value structure
- Tests input validation and error handling
- Ensures all required fields are present in the output

### 2. Disparity-Based Detection (`TestDisparityBasedDetection`)
- Tests horizontal disparity pattern analysis
- Validates handling of positive/negative disparities
- Checks disparity range and distribution metrics

### 3. Occlusion Analysis (`TestOcclusionAnalysis`)
- Tests detection of occlusion violations
- Validates left/right occlusion consistency
- Checks occlusion patterns in swapped views

### 4. Feature Matching Criteria (`TestFeatureMatchingCriteria`)
- Tests feature correspondence analysis
- Validates epipolar constraint satisfaction
- Checks fundamental matrix estimation errors

### 5. Depth Map Validation (`TestDepthMapValidation`)
- Tests depth map consistency metrics
- Validates depth smoothness and edge alignment
- Checks handling of invalid depth regions

### 6. Geometric Constraints (`TestGeometricConstraints`)
- Tests rectification quality impact
- Validates vertical disparity detection
- Checks geometric consistency metrics

### 7. Robustness and Edge Cases (`TestRobustnessAndEdgeCases`)
- Tests performance on low-texture regions
- Validates noise robustness
- Checks extreme disparity handling

### 8. Temporal Consistency (`TestTemporalConsistency`)
- Tests stability across video frames
- Validates scene change handling
- Checks temporal filtering effectiveness

### 9. Performance Metrics (`TestPerformanceMetrics`)
- Tests confidence statistics computation
- Validates computational efficiency metrics
- Checks processing time reporting

### 10. Integration Scenarios (`TestIntegrationScenarios`)
- Tests VR180 format handling
- Validates full pipeline on correct stereo
- Tests complete detection on swapped views

## Expected Algorithm Interface

The algorithm should accept two numpy arrays (left and right eye images) and return a dictionary with:

```python
{
    'mismatch_score': float,  # [0, inf) - higher indicates more likely mismatch
    'is_mismatch': bool,      # Binary decision
    'criteria_scores': {      # Individual detection criteria scores
        'disparity_consistency': float,
        'occlusion_analysis': float,
        'feature_matching': float,
        'depth_map_validity': float,
        'geometric_constraints': float,
        # ... additional criteria
    },
    'weights_used': dict,     # Weights applied to each criterion
    'disparity_quality': {    # Disparity estimation quality metrics
        'mean_disparity': float,
        'disparity_range': tuple,
        'invalid_pixels_ratio': float,
        'confidence_mean': float
    },
    'confidence_stats': {     # Detection confidence statistics
        'high_confidence_ratio': float,
        'low_confidence_ratio': float,
        'median_confidence': float
    },
    'error': str  # Optional error message if detection failed
}
```

## Usage

### Running All Tests
```bash
pytest test_channel_mismatch_detection.py -v
```

### Running Specific Test Categories
```bash
# Test only disparity-based detection
pytest test_channel_mismatch_detection.py::TestDisparityBasedDetection -v

# Test only robustness
pytest test_channel_mismatch_detection.py::TestRobustnessAndEdgeCases -v
```

### Integration with Your Algorithm

Replace the mock `detect_channel_mismatch` method with your actual implementation:

```python
def detect_channel_mismatch(left_image: np.ndarray, right_image: np.ndarray) -> Dict[str, Any]:
    # Your algorithm implementation here
    return results
```

## Evaluation Metrics

The test suite evaluates algorithms based on:

1. **Detection Accuracy** - Correctly identifying swapped vs. correct stereo pairs
2. **Robustness** - Performance under challenging conditions (noise, low texture)
3. **Temporal Stability** - Consistent detection across video frames
4. **Computational Efficiency** - Processing time and resource usage
5. **Confidence Reliability** - Accuracy of confidence estimates

## References

This test suite is based on research from:
- MSU Graphics & Media Lab Video Group (VQMT3D project)
- Publications on stereoscopic video quality assessment
- Industry standards for VR/3D video production

Key artifacts that indicate channel mismatch:
- Inverted depth perception
- Occlusion violations
- Epipolar constraint violations
- Unusual disparity distributions
- High vertical disparity presence

## Requirements

- Python 3.7+
- numpy
- pytest
- opencv-python (optional, for image loading)
- unittest.mock (for test mocking)

## Contributing

When adding new tests, ensure they:
1. Follow the existing test structure
2. Include clear documentation
3. Test specific aspects of channel mismatch detection
4. Use realistic parameter values based on actual VR video characteristics

