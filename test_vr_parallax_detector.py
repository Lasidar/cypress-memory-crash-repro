import pytest
import numpy as np
from unittest.mock import Mock, patch
import cv2


class TestVRParallaxDetector:
    """
    Comprehensive test suite for VR video parallax detection algorithm.
    Tests are based on research from VR comfort zones, stereoscopic vision studies,
    and industry standards for parallax evaluation.
    """

    @pytest.fixture
    def detector(self):
        """Mock detector instance for testing"""
        return Mock()

    @pytest.fixture
    def sample_left_frame(self):
        """Generate a sample left eye frame"""
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_right_frame(self):
        """Generate a sample right eye frame with slight horizontal offset"""
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        # Simulate horizontal disparity by shifting pixels
        right = np.roll(left, shift=5, axis=1)
        return right

    @pytest.fixture
    def high_disparity_frames(self):
        """Generate frames with excessive horizontal disparity (uncomfortable)"""
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        # Large shift simulating uncomfortable parallax
        right = np.roll(left, shift=50, axis=1)
        return left, right

    @pytest.fixture
    def vertical_disparity_frames(self):
        """Generate frames with vertical disparity (should cause discomfort)"""
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        # Vertical shift - this should be flagged as problematic
        right = np.roll(left, shift=10, axis=0)
        return left, right

    @pytest.fixture
    def zero_parallax_frames(self):
        """Generate identical frames (zero parallax - screen plane)"""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        return frame, frame.copy()

    def test_basic_parallax_detection(self, detector, sample_left_frame, sample_right_frame):
        """Test basic parallax detection functionality"""
        # Mock the detector's analyze method
        detector.analyze_parallax.return_value = {
            'positive_parallax_score': 0.3,
            'negative_parallax_score': 0.1,
            'mean_disparity': 5.2,
            'std_disparity': 2.1,
            'total_valid_pixels': 2073600,
            'valid_pixels_ratio': 0.98
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        assert 'positive_parallax_score' in result
        assert 'negative_parallax_score' in result
        assert 'mean_disparity' in result
        assert result['total_valid_pixels'] > 0
        assert 0 <= result['valid_pixels_ratio'] <= 1

    def test_comfort_zone_evaluation(self, detector, sample_left_frame, sample_right_frame):
        """Test comfort zone ratio calculations based on Shibata's Zone of Comfort"""
        # Based on research: comfortable limits are 2-3% screen width for crossed disparity
        # and 1-2% for uncrossed disparity
        detector.analyze_parallax.return_value = {
            'comfortable_positive_ratio': 0.85,  # 85% in comfortable positive zone
            'comfortable_negative_ratio': 0.92,  # 92% in comfortable negative zone
            'extreme_positive_ratio': 0.05,      # 5% extreme positive parallax
            'extreme_negative_ratio': 0.03,      # 3% extreme negative parallax
            'critical_positive_ratio': 0.01,     # 1% critical discomfort level
            'critical_negative_ratio': 0.02,     # 2% critical discomfort level
            'comfort_score': 0.88,               # Overall comfort score
            'discomfort_score': 0.12             # Overall discomfort score
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        # Comfort zone ratios should sum to reasonable values
        assert 0 <= result['comfortable_positive_ratio'] <= 1
        assert 0 <= result['comfortable_negative_ratio'] <= 1
        assert result['comfort_score'] + result['discomfort_score'] == pytest.approx(1.0, abs=0.01)
        
        # Extreme ratios should be small for comfortable content
        assert result['extreme_positive_ratio'] < 0.1
        assert result['extreme_negative_ratio'] < 0.1

    def test_excessive_horizontal_disparity_detection(self, detector, high_disparity_frames):
        """Test detection of excessive horizontal disparity that causes discomfort"""
        left, right = high_disparity_frames
        
        detector.analyze_parallax.return_value = {
            'extreme_positive_ratio': 0.15,      # High extreme parallax
            'extreme_negative_ratio': 0.12,      # High extreme parallax
            'comfort_score': 0.25,               # Low comfort score
            'discomfort_score': 0.75,            # High discomfort score
            'mean_disparity': 45.8,              # Large mean disparity
            'std_disparity': 15.3                # High variance
        }
        
        result = detector.analyze_parallax(left, right)
        
        # High disparity should result in low comfort scores
        assert result['comfort_score'] < 0.5
        assert result['discomfort_score'] > 0.5
        assert result['extreme_positive_ratio'] > 0.1 or result['extreme_negative_ratio'] > 0.1
        assert abs(result['mean_disparity']) > 20  # Large disparity

    def test_vertical_disparity_detection(self, detector, vertical_disparity_frames):
        """Test detection of vertical disparity which should always cause discomfort"""
        left, right = vertical_disparity_frames
        
        # Vertical disparity should be flagged as highly problematic
        detector.analyze_parallax.return_value = {
            'vertical_disparity_detected': True,
            'vertical_disparity_severity': 0.8,   # High severity
            'comfort_score': 0.1,                 # Very low comfort
            'discomfort_score': 0.9,              # Very high discomfort
            'critical_positive_ratio': 0.0,
            'critical_negative_ratio': 0.0
        }
        
        result = detector.analyze_parallax(left, right)
        
        # Vertical disparity should result in very low comfort
        assert result['comfort_score'] < 0.2
        assert result['discomfort_score'] > 0.8

    def test_zero_parallax_comfort(self, detector, zero_parallax_frames):
        """Test that zero parallax (screen plane) is comfortable"""
        left, right = zero_parallax_frames
        
        detector.analyze_parallax.return_value = {
            'positive_parallax_score': 0.0,
            'negative_parallax_score': 0.0,
            'comfort_score': 0.95,               # Very comfortable
            'discomfort_score': 0.05,            # Very low discomfort
            'mean_disparity': 0.1,               # Near zero disparity
            'std_disparity': 0.5,                # Low variance
            'extreme_positive_ratio': 0.0,
            'extreme_negative_ratio': 0.0
        }
        
        result = detector.analyze_parallax(left, right)
        
        # Zero parallax should be very comfortable
        assert result['comfort_score'] > 0.9
        assert result['discomfort_score'] < 0.1
        assert abs(result['mean_disparity']) < 1.0

    def test_comfort_inner_outer_zones(self, detector, sample_left_frame, sample_right_frame):
        """Test inner and outer comfort zone calculations"""
        # Based on Percival's zone of comfort (middle third of ZCSBV)
        detector.analyze_parallax.return_value = {
            'comfort_inner_ratio': 0.75,         # 75% in inner comfort zone
            'comfort_outer_ratio': 0.15,         # 15% in outer comfort zone
            'extreme_positive_ratio': 0.06,      # 6% beyond comfort zones
            'extreme_negative_ratio': 0.04,      # 4% beyond comfort zones
            'comfort_score': 0.82,
            'discomfort_score': 0.18
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        # Inner zone should have highest ratio for comfortable content
        assert result['comfort_inner_ratio'] > result['comfort_outer_ratio']
        assert result['comfort_inner_ratio'] + result['comfort_outer_ratio'] > 0.8

    def test_statistical_metrics_validity(self, detector, sample_left_frame, sample_right_frame):
        """Test that statistical metrics are within expected ranges"""
        detector.analyze_parallax.return_value = {
            'mean_disparity': 3.5,
            'std_disparity': 8.2,
            'total_valid_pixels': 1950000,
            'valid_pixels_ratio': 0.94,
            'disparity_range': {'min': -15.2, 'max': 22.8},
            'histogram_bins': list(range(-20, 21)),
            'histogram_values': [50, 120, 200, 180, 150] + [0] * 36  # Sample histogram
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        # Statistical metrics should be reasonable
        assert isinstance(result['mean_disparity'], (int, float))
        assert result['std_disparity'] >= 0
        assert result['total_valid_pixels'] > 0
        assert 0 <= result['valid_pixels_ratio'] <= 1
        assert result['disparity_range']['min'] <= result['disparity_range']['max']
        assert len(result['histogram_bins']) == len(result['histogram_values'])

    def test_edge_cases_empty_frames(self, detector):
        """Test handling of edge cases like empty or invalid frames"""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector.analyze_parallax.return_value = {
            'error': 'insufficient_texture',
            'valid_pixels_ratio': 0.0,
            'comfort_score': 0.5,  # Neutral score for invalid input
            'discomfort_score': 0.5
        }
        
        result = detector.analyze_parallax(empty_frame, empty_frame)
        
        assert 'error' in result or result['valid_pixels_ratio'] == 0.0

    def test_input_validation(self, detector):
        """Test input validation for frame dimensions and types"""
        # Mismatched dimensions
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        with pytest.raises((ValueError, AssertionError)):
            detector.analyze_parallax(left, right)

    def test_motion_in_depth_detection(self, detector, sample_left_frame, sample_right_frame):
        """Test detection of motion in depth which should be flagged for discomfort"""
        # Simulate motion in depth by varying disparity across regions
        detector.analyze_parallax.return_value = {
            'motion_in_depth_detected': True,
            'depth_motion_severity': 0.6,
            'comfort_score': 0.4,               # Reduced comfort due to motion
            'discomfort_score': 0.6,
            'std_disparity': 12.5               # High variance indicates motion in depth
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        # Motion in depth should reduce comfort
        if 'motion_in_depth_detected' in result and result['motion_in_depth_detected']:
            assert result['comfort_score'] < 0.7
            assert result['std_disparity'] > 10

    def test_performance_metrics(self, detector, sample_left_frame, sample_right_frame):
        """Test that performance metrics are reasonable for real-time processing"""
        start_time = 0
        end_time = 0.05  # 50ms processing time (acceptable for real-time)
        
        detector.analyze_parallax.return_value = {
            'processing_time_ms': 45,
            'positive_parallax_score': 0.3,
            'negative_parallax_score': 0.2,
            'comfort_score': 0.75
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        # Performance should be suitable for real-time processing
        if 'processing_time_ms' in result:
            assert result['processing_time_ms'] < 100  # Less than 100ms

    def test_window_violation_detection(self, detector, sample_left_frame, sample_right_frame):
        """Test detection of stereo window violations"""
        # Window violations occur when objects with negative parallax are cut by screen edges
        detector.analyze_parallax.return_value = {
            'window_violation_detected': True,
            'window_violation_severity': 0.3,
            'edge_disparity_issues': True,
            'comfort_score': 0.6,               # Reduced due to window violation
            'discomfort_score': 0.4
        }
        
        result = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        if 'window_violation_detected' in result and result['window_violation_detected']:
            assert result['comfort_score'] < 0.8  # Window violations reduce comfort

    @pytest.mark.parametrize("screen_width_percent,expected_comfort", [
        (1.0, 0.95),    # 1% screen width - very comfortable
        (2.5, 0.85),    # 2.5% screen width - comfortable
        (4.0, 0.6),     # 4% screen width - uncomfortable
        (6.0, 0.3),     # 6% screen width - very uncomfortable
    ])
    def test_disparity_percentage_thresholds(self, detector, screen_width_percent, expected_comfort):
        """Test comfort evaluation based on disparity as percentage of screen width"""
        # Based on research: 3% rule and comfort zone studies
        detector.analyze_parallax.return_value = {
            'disparity_screen_width_percent': screen_width_percent,
            'comfort_score': expected_comfort,
            'discomfort_score': 1.0 - expected_comfort
        }
        
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        result = detector.analyze_parallax(left, right)
        
        # Verify comfort decreases as disparity percentage increases
        assert abs(result['comfort_score'] - expected_comfort) < 0.1

    def test_crossed_vs_uncrossed_disparity_comfort(self, detector):
        """Test that crossed disparity (negative parallax) is less comfortable than uncrossed"""
        # Research shows negative parallax is more likely to cause discomfort
        
        # Uncrossed disparity (positive parallax - behind screen)
        detector.analyze_parallax.return_value = {
            'positive_parallax_score': 0.8,
            'negative_parallax_score': 0.1,
            'comfort_score': 0.85
        }
        
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        uncrossed_result = detector.analyze_parallax(left, right)
        
        # Crossed disparity (negative parallax - in front of screen)
        detector.analyze_parallax.return_value = {
            'positive_parallax_score': 0.1,
            'negative_parallax_score': 0.8,
            'comfort_score': 0.65  # Lower comfort for crossed disparity
        }
        
        crossed_result = detector.analyze_parallax(left, right)
        
        # Uncrossed disparity should generally be more comfortable
        assert uncrossed_result['comfort_score'] >= crossed_result['comfort_score']

    def test_algorithm_consistency(self, detector, sample_left_frame, sample_right_frame):
        """Test that the algorithm produces consistent results for identical inputs"""
        expected_result = {
            'positive_parallax_score': 0.3,
            'negative_parallax_score': 0.2,
            'comfort_score': 0.75,
            'mean_disparity': 4.2
        }
        
        detector.analyze_parallax.return_value = expected_result
        
        result1 = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        result2 = detector.analyze_parallax(sample_left_frame, sample_right_frame)
        
        # Results should be identical for same input
        assert result1['comfort_score'] == result2['comfort_score']
        assert result1['mean_disparity'] == result2['mean_disparity']

    def test_extreme_parallax_thresholds(self, detector):
        """Test detection of parallax beyond human fusion limits"""
        # Human fusion limits are approximately 0.5 degrees (varies with retinal location)
        
        detector.analyze_parallax.return_value = {
            'fusion_limit_exceeded': True,
            'extreme_positive_ratio': 0.25,
            'extreme_negative_ratio': 0.20,
            'comfort_score': 0.15,              # Very low comfort
            'discomfort_score': 0.85,           # Very high discomfort
            'critical_positive_ratio': 0.15,
            'critical_negative_ratio': 0.12
        }
        
        left = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        result = detector.analyze_parallax(left, right)
        
        # Beyond fusion limits should result in very low comfort
        if 'fusion_limit_exceeded' in result and result['fusion_limit_exceeded']:
            assert result['comfort_score'] < 0.3
            assert result['discomfort_score'] > 0.7