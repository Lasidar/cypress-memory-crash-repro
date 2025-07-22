"""
Utility functions and helpers for VR disparity detection testing.

This module provides additional utilities for creating test data, evaluating results,
and generating reports for VR disparity detection algorithms.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import confusion_matrix
import json
import os


@dataclass
class VRCameraParameters:
    """Parameters for VR camera configuration."""
    baseline: float = 65.0  # mm, interpupillary distance
    focal_length: float = 500.0  # pixels
    sensor_width: float = 36.0  # mm
    sensor_height: float = 24.0  # mm
    resolution: Tuple[int, int] = (1920, 1080)
    field_of_view: float = 90.0  # degrees
    lens_distortion_k1: float = 0.0
    lens_distortion_k2: float = 0.0
    convergence_distance: float = 2000.0  # mm


@dataclass
class DisparityGroundTruth:
    """Container for ground truth disparity data with metadata."""
    disparity_map: np.ndarray
    depth_map: Optional[np.ndarray] = None
    occlusion_mask: Optional[np.ndarray] = None
    object_mask: Optional[np.ndarray] = None
    material_properties: Optional[Dict[int, str]] = None
    confidence_map: Optional[np.ndarray] = None


class AdvancedStereoGenerator:
    """Advanced synthetic stereo pair generator with realistic effects."""
    
    def __init__(self, camera_params: VRCameraParameters):
        self.camera_params = camera_params
    
    def create_realistic_scene(
        self,
        scene_type: str = "indoor",
        lighting: str = "normal",
        add_noise: bool = True,
        add_compression_artifacts: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, DisparityGroundTruth]:
        """
        Create a realistic synthetic scene with various effects.
        
        Args:
            scene_type: Type of scene ('indoor', 'outdoor', 'mixed')
            lighting: Lighting condition ('normal', 'low', 'high_contrast')
            add_noise: Whether to add sensor noise
            add_compression_artifacts: Whether to simulate compression
            
        Returns:
            Tuple of (left_image, right_image, ground_truth)
        """
        height, width = self.camera_params.resolution[::-1]
        
        # Create base scene
        if scene_type == "indoor":
            scene = self._create_indoor_scene(width, height)
        elif scene_type == "outdoor":
            scene = self._create_outdoor_scene(width, height)
        else:
            scene = self._create_mixed_scene(width, height)
        
        # Apply lighting
        scene = self._apply_lighting(scene, lighting)
        
        # Generate disparity and create stereo pair
        left_image, right_image, ground_truth = self._create_stereo_from_scene(scene)
        
        # Add realistic effects
        if add_noise:
            left_image = self._add_sensor_noise(left_image)
            right_image = self._add_sensor_noise(right_image)
        
        if add_compression_artifacts:
            left_image = self._add_compression_artifacts(left_image)
            right_image = self._add_compression_artifacts(right_image)
        
        return left_image, right_image, ground_truth
    
    def _create_indoor_scene(self, width: int, height: int) -> Dict:
        """Create an indoor scene with walls, floor, and objects."""
        scene = {
            'layers': [],
            'depths': [],
            'materials': []
        }
        
        # Background wall
        wall = np.ones((height, width, 3), dtype=np.uint8) * 180
        wall_depth = 3000.0  # mm
        scene['layers'].append(wall)
        scene['depths'].append(wall_depth)
        scene['materials'].append('diffuse')
        
        # Add furniture-like objects
        # Table
        table_h, table_w = height // 3, width // 2
        table_y, table_x = 2 * height // 3, width // 4
        table = np.zeros((height, width, 3), dtype=np.uint8)
        table[table_y:table_y+table_h, table_x:table_x+table_w] = [139, 69, 19]  # Brown
        table_mask = np.zeros((height, width), dtype=bool)
        table_mask[table_y:table_y+table_h, table_x:table_x+table_w] = True
        table_depth = 1500.0  # mm
        scene['layers'].append((table, table_mask))
        scene['depths'].append(table_depth)
        scene['materials'].append('wood')
        
        # Add a specular object (TV/monitor)
        tv_h, tv_w = height // 4, width // 3
        tv_y, tv_x = height // 4, width // 2
        tv = np.zeros((height, width, 3), dtype=np.uint8)
        tv[tv_y:tv_y+tv_h, tv_x:tv_x+tv_w] = [20, 20, 20]  # Dark gray
        tv_mask = np.zeros((height, width), dtype=bool)
        tv_mask[tv_y:tv_y+tv_h, tv_x:tv_x+tv_w] = True
        tv_depth = 2000.0  # mm
        scene['layers'].append((tv, tv_mask))
        scene['depths'].append(tv_depth)
        scene['materials'].append('specular')
        
        return scene
    
    def _create_outdoor_scene(self, width: int, height: int) -> Dict:
        """Create an outdoor scene with sky, ground, and objects."""
        scene = {
            'layers': [],
            'depths': [],
            'materials': []
        }
        
        # Sky gradient
        sky = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height // 2):
            color_value = int(135 + (255 - 135) * y / (height // 2))
            sky[y, :] = [color_value, color_value, 255]  # Blue gradient
        sky_depth = 10000.0  # mm (far)
        scene['layers'].append(sky)
        scene['depths'].append(sky_depth)
        scene['materials'].append('sky')
        
        # Ground
        ground = np.zeros((height, width, 3), dtype=np.uint8)
        ground[height // 2:, :] = [34, 139, 34]  # Green
        ground_mask = np.zeros((height, width), dtype=bool)
        ground_mask[height // 2:, :] = True
        ground_depth = np.linspace(1000, 5000, height // 2)  # Receding ground
        scene['layers'].append((ground, ground_mask))
        scene['depths'].append(ground_depth)
        scene['materials'].append('grass')
        
        # Add trees
        for i in range(3):
            tree_x = (i + 1) * width // 4
            tree_h = height // 3
            tree_w = width // 8
            tree_y = height // 2 - tree_h // 2
            
            tree = np.zeros((height, width, 3), dtype=np.uint8)
            # Trunk
            trunk_w = tree_w // 3
            trunk_x = tree_x - trunk_w // 2
            tree[tree_y + tree_h // 2:height // 2, trunk_x:trunk_x + trunk_w] = [139, 69, 19]
            # Leaves
            cv2.ellipse(tree, (tree_x, tree_y + tree_h // 3), 
                       (tree_w // 2, tree_h // 3), 0, 0, 360, (0, 128, 0), -1)
            
            tree_mask = np.any(tree > 0, axis=2)
            tree_depth = 1500 + i * 1000  # mm
            scene['layers'].append((tree, tree_mask))
            scene['depths'].append(tree_depth)
            scene['materials'].append('vegetation')
        
        return scene
    
    def _create_mixed_scene(self, width: int, height: int) -> Dict:
        """Create a mixed indoor/outdoor scene."""
        # Combine elements from both
        indoor = self._create_indoor_scene(width, height)
        outdoor = self._create_outdoor_scene(width, height)
        
        # Merge scenes (simplified)
        scene = {
            'layers': indoor['layers'][:2] + outdoor['layers'][2:],
            'depths': indoor['depths'][:2] + outdoor['depths'][2:],
            'materials': indoor['materials'][:2] + outdoor['materials'][2:]
        }
        
        return scene
    
    def _apply_lighting(self, scene: Dict, lighting: str) -> Dict:
        """Apply lighting effects to the scene."""
        if lighting == "low":
            # Reduce overall brightness
            for i, layer in enumerate(scene['layers']):
                if isinstance(layer, tuple):
                    img, mask = layer
                    img = (img * 0.3).astype(np.uint8)
                    scene['layers'][i] = (img, mask)
                else:
                    scene['layers'][i] = (layer * 0.3).astype(np.uint8)
        
        elif lighting == "high_contrast":
            # Add strong directional lighting
            for i, layer in enumerate(scene['layers']):
                if isinstance(layer, tuple):
                    img, mask = layer
                else:
                    img = layer
                    mask = np.ones(img.shape[:2], dtype=bool)
                
                # Create lighting gradient
                light_grad = np.linspace(0.5, 1.5, img.shape[1])
                for c in range(3):
                    img[:, :, c] = np.clip(img[:, :, c] * light_grad, 0, 255).astype(np.uint8)
                
                if isinstance(layer, tuple):
                    scene['layers'][i] = (img, mask)
                else:
                    scene['layers'][i] = img
        
        return scene
    
    def _create_stereo_from_scene(
        self, 
        scene: Dict
    ) -> Tuple[np.ndarray, np.ndarray, DisparityGroundTruth]:
        """Generate stereo pair from scene description."""
        height, width = self.camera_params.resolution[::-1]
        
        # Initialize images and ground truth
        left_image = np.zeros((height, width, 3), dtype=np.uint8)
        right_image = np.zeros((height, width, 3), dtype=np.uint8)
        depth_map = np.full((height, width), np.inf)
        object_mask = np.zeros((height, width), dtype=np.int32)
        material_map = {}
        
        # Composite layers from back to front
        for layer_idx, (layer, depth, material) in enumerate(
            zip(scene['layers'], scene['depths'], scene['materials'])
        ):
            
            if isinstance(layer, tuple):
                img, mask = layer
            else:
                img = layer
                mask = np.ones(img.shape[:2], dtype=bool)
            
            # Calculate disparity
            if isinstance(depth, np.ndarray):
                # Variable depth (e.g., ground plane)
                disparity = (self.camera_params.baseline * self.camera_params.focal_length) / depth
            else:
                # Constant depth
                disparity = (self.camera_params.baseline * self.camera_params.focal_length) / depth
                disparity = np.full(mask.shape, disparity)
            
            # Update depth map where this layer is visible
            layer_visible = mask & (depth < depth_map)
            if isinstance(depth, np.ndarray):
                depth_map[layer_visible] = depth[layer_visible]
            else:
                depth_map[layer_visible] = depth
            
            # Update object mask
            object_mask[layer_visible] = layer_idx + 1
            material_map[layer_idx + 1] = material
            
            # Composite into left image
            left_image[mask] = img[mask]
            
            # Create right image with disparity shift
            for y in range(height):
                for x in range(width):
                    if mask[y, x]:
                        x_right = x - int(disparity[y, x] if isinstance(disparity, np.ndarray) else disparity)
                        if 0 <= x_right < width:
                            right_image[y, x_right] = img[y, x]
        
        # Convert depth to disparity for ground truth
        disparity_map = (self.camera_params.baseline * self.camera_params.focal_length) / depth_map
        disparity_map = np.clip(disparity_map, 0, 255).astype(np.uint8)
        
        # Create occlusion mask (pixels visible in left but not right)
        occlusion_mask = np.zeros((height, width), dtype=bool)
        for y in range(height):
            for x in range(width):
                if left_image[y, x].any() and not right_image[y, x].any():
                    occlusion_mask[y, x] = True
        
        ground_truth = DisparityGroundTruth(
            disparity_map=disparity_map,
            depth_map=depth_map,
            occlusion_mask=occlusion_mask,
            object_mask=object_mask,
            material_properties=material_map
        )
        
        return left_image, right_image, ground_truth
    
    def _add_sensor_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic sensor noise to image."""
        # Gaussian noise
        noise = np.random.normal(0, 2, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Salt and pepper noise (dead pixels)
        salt_pepper = np.random.random(image.shape[:2])
        noisy[salt_pepper < 0.0005] = 0  # Dead pixels
        noisy[salt_pepper > 0.9995] = 255  # Hot pixels
        
        return noisy
    
    def _add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        # Encode and decode with JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        compressed = cv2.imdecode(encimg, 1)
        
        return compressed


class DisparityVisualization:
    """Utilities for visualizing disparity maps and evaluation results."""
    
    @staticmethod
    def create_disparity_colormap(
        disparity: np.ndarray,
        colormap: str = 'jet',
        min_disp: Optional[float] = None,
        max_disp: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert disparity map to color visualization.
        
        Args:
            disparity: Disparity map
            colormap: OpenCV colormap name
            min_disp: Minimum disparity for normalization
            max_disp: Maximum disparity for normalization
            
        Returns:
            Color-mapped disparity image
        """
        if min_disp is None:
            min_disp = np.min(disparity[disparity > 0]) if np.any(disparity > 0) else 0
        if max_disp is None:
            max_disp = np.max(disparity)
        
        # Normalize to 0-255
        normalized = np.clip((disparity - min_disp) / (max_disp - min_disp) * 255, 0, 255)
        normalized = normalized.astype(np.uint8)
        
        # Apply colormap
        cmap = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(normalized, cmap)
        
        # Set invalid disparities to black
        colored[disparity == 0] = 0
        
        return colored
    
    @staticmethod
    def create_error_visualization(
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Create error visualization showing correct/incorrect pixels.
        
        Args:
            estimated: Estimated disparity map
            ground_truth: Ground truth disparity map
            threshold: Error threshold for bad pixels
            
        Returns:
            Error visualization (green=correct, red=bad, gray=unknown)
        """
        height, width = estimated.shape
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate error
        error = np.abs(estimated.astype(float) - ground_truth.astype(float))
        
        # Valid mask (where ground truth is known)
        valid_mask = ground_truth > 0
        
        # Classify pixels
        correct = valid_mask & (error <= threshold)
        incorrect = valid_mask & (error > threshold)
        unknown = ~valid_mask
        
        # Color code
        vis[correct] = [0, 255, 0]  # Green for correct
        vis[incorrect] = [0, 0, 255]  # Red for incorrect
        vis[unknown] = [128, 128, 128]  # Gray for unknown
        
        return vis
    
    @staticmethod
    def create_confidence_visualization(
        confidence_map: np.ndarray,
        overlay_on: Optional[np.ndarray] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize confidence map, optionally overlaid on image.
        
        Args:
            confidence_map: Confidence values [0, 1]
            overlay_on: Optional image to overlay on
            alpha: Blending factor
            
        Returns:
            Confidence visualization
        """
        # Normalize confidence to 0-255
        conf_vis = (confidence_map * 255).astype(np.uint8)
        
        # Apply colormap (red=low confidence, green=high confidence)
        conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_RDYlGn)
        
        if overlay_on is not None:
            # Blend with original image
            result = cv2.addWeighted(overlay_on, 1-alpha, conf_colored, alpha, 0)
            return result
        
        return conf_colored
    
    @staticmethod
    def plot_disparity_histogram(
        disparity: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        bins: int = 50,
        save_path: Optional[str] = None
    ):
        """Plot histogram of disparity values."""
        plt.figure(figsize=(10, 6))
        
        # Plot estimated disparity
        valid_disp = disparity[disparity > 0].flatten()
        plt.hist(valid_disp, bins=bins, alpha=0.7, label='Estimated', density=True)
        
        # Plot ground truth if available
        if ground_truth is not None:
            valid_gt = ground_truth[ground_truth > 0].flatten()
            plt.hist(valid_gt, bins=bins, alpha=0.7, label='Ground Truth', density=True)
        
        plt.xlabel('Disparity (pixels)')
        plt.ylabel('Density')
        plt.title('Disparity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_evaluation_report(
        results: Dict[str, float],
        save_path: str,
        include_plots: bool = True
    ):
        """
        Create a comprehensive evaluation report.
        
        Args:
            results: Dictionary of evaluation metrics
            save_path: Path to save report
            include_plots: Whether to include visualizations
        """
        report = {
            'timestamp': str(np.datetime64('now')),
            'metrics': results,
            'summary': {
                'overall_quality': 'good' if results.get('bad_pixel_error', 100) < 10 else 'poor',
                'key_findings': []
            }
        }
        
        # Analyze results
        if results.get('bad_pixel_error', 100) > 20:
            report['summary']['key_findings'].append('High bad pixel percentage indicates poor accuracy')
        
        if results.get('edge_preservation', 0) < 0.7:
            report['summary']['key_findings'].append('Poor edge preservation detected')
        
        if results.get('temporal_consistency', 0) < 0.8:
            report['summary']['key_findings'].append('Temporal inconsistency detected')
        
        # Save JSON report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


class VRComfortAnalyzer:
    """Analyze VR viewing comfort based on disparity characteristics."""
    
    def __init__(self, camera_params: VRCameraParameters):
        self.camera_params = camera_params
    
    def analyze_comfort(
        self,
        disparity_map: np.ndarray,
        frame_rate: float = 90.0  # Hz
    ) -> Dict[str, float]:
        """
        Analyze viewing comfort based on disparity characteristics.
        
        Args:
            disparity_map: Disparity map to analyze
            frame_rate: Display frame rate
            
        Returns:
            Dictionary of comfort metrics
        """
        results = {}
        
        # Maximum disparity (affects convergence)
        max_disp = np.max(disparity_map)
        results['max_disparity'] = float(max_disp)
        
        # Calculate vergence angle
        vergence_angle = np.arctan(max_disp * self.camera_params.sensor_width / 
                                  (self.camera_params.resolution[0] * self.camera_params.focal_length))
        results['max_vergence_angle_deg'] = float(np.degrees(vergence_angle))
        
        # Disparity gradient (rapid depth changes)
        grad_x = np.abs(np.gradient(disparity_map, axis=1))
        grad_y = np.abs(np.gradient(disparity_map, axis=0))
        max_gradient = np.max(np.sqrt(grad_x**2 + grad_y**2))
        results['max_disparity_gradient'] = float(max_gradient)
        
        # Screen disparity in mm (for comfort assessment)
        pixel_pitch = self.camera_params.sensor_width / self.camera_params.resolution[0]
        screen_disparity_mm = max_disp * pixel_pitch
        results['max_screen_disparity_mm'] = float(screen_disparity_mm)
        
        # Comfort scores
        # Based on research: comfortable viewing typically < 1 degree vergence
        vergence_comfort = 1.0 - min(results['max_vergence_angle_deg'] / 2.0, 1.0)
        results['vergence_comfort_score'] = float(vergence_comfort)
        
        # Gradient comfort (rapid changes cause discomfort)
        gradient_comfort = 1.0 - min(max_gradient / 10.0, 1.0)
        results['gradient_comfort_score'] = float(gradient_comfort)
        
        # Overall comfort score
        results['overall_comfort_score'] = float(
            0.6 * vergence_comfort + 0.4 * gradient_comfort
        )
        
        # Recommendations
        results['recommendations'] = []
        if results['max_vergence_angle_deg'] > 1.5:
            results['recommendations'].append('Reduce maximum disparity for comfort')
        if max_gradient > 5:
            results['recommendations'].append('Smooth depth transitions to reduce eye strain')
        
        return results
    
    def analyze_temporal_comfort(
        self,
        disparity_sequence: List[np.ndarray],
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Analyze temporal comfort across a sequence of frames.
        
        Args:
            disparity_sequence: List of disparity maps
            timestamps: Optional timestamps for each frame
            
        Returns:
            Dictionary of temporal comfort metrics
        """
        if len(disparity_sequence) < 2:
            return {'error': 'Need at least 2 frames for temporal analysis'}
        
        results = {}
        
        # Calculate frame-to-frame disparity changes
        temporal_changes = []
        for i in range(1, len(disparity_sequence)):
            diff = np.abs(disparity_sequence[i] - disparity_sequence[i-1])
            temporal_changes.append(np.mean(diff))
        
        results['mean_temporal_change'] = float(np.mean(temporal_changes))
        results['max_temporal_change'] = float(np.max(temporal_changes))
        
        # Flicker detection (rapid oscillations)
        if len(temporal_changes) > 2:
            # Check for alternating changes
            oscillations = 0
            for i in range(1, len(temporal_changes)):
                if temporal_changes[i] * temporal_changes[i-1] < 0:  # Sign change
                    oscillations += 1
            
            results['oscillation_count'] = oscillations
            results['flicker_risk'] = 'high' if oscillations > len(temporal_changes) * 0.3 else 'low'
        
        # Comfort score based on temporal stability
        temporal_comfort = 1.0 - min(results['mean_temporal_change'] / 5.0, 1.0)
        results['temporal_comfort_score'] = float(temporal_comfort)
        
        return results


def generate_test_report(
    test_results: Dict[str, Dict[str, float]],
    output_dir: str,
    algorithm_name: str = "Unknown"
):
    """
    Generate a comprehensive test report with visualizations.
    
    Args:
        test_results: Dictionary of test results
        output_dir: Directory to save report
        algorithm_name: Name of the algorithm being tested
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary
    summary = {
        'algorithm': algorithm_name,
        'timestamp': str(np.datetime64('now')),
        'overall_results': {},
        'test_details': test_results
    }
    
    # Calculate overall metrics
    all_bad_pixel_errors = []
    all_mae_values = []
    all_comfort_scores = []
    
    for test_name, metrics in test_results.items():
        if 'bad_pixel_error' in metrics:
            all_bad_pixel_errors.append(metrics['bad_pixel_error'])
        if 'mean_absolute_error' in metrics:
            all_mae_values.append(metrics['mean_absolute_error'])
        if 'overall_comfort_score' in metrics:
            all_comfort_scores.append(metrics['overall_comfort_score'])
    
    if all_bad_pixel_errors:
        summary['overall_results']['mean_bad_pixel_error'] = float(np.mean(all_bad_pixel_errors))
        summary['overall_results']['std_bad_pixel_error'] = float(np.std(all_bad_pixel_errors))
    
    if all_mae_values:
        summary['overall_results']['mean_mae'] = float(np.mean(all_mae_values))
    
    if all_comfort_scores:
        summary['overall_results']['mean_comfort_score'] = float(np.mean(all_comfort_scores))
    
    # Save summary
    summary_path = os.path.join(output_dir, 'test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    if all_bad_pixel_errors:
        plt.figure(figsize=(10, 6))
        test_names = list(test_results.keys())
        plt.bar(range(len(all_bad_pixel_errors)), all_bad_pixel_errors)
        plt.xticks(range(len(all_bad_pixel_errors)), test_names, rotation=45, ha='right')
        plt.ylabel('Bad Pixel Error (%)')
        plt.title(f'Bad Pixel Error by Test - {algorithm_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bad_pixel_errors.png'), dpi=150)
        plt.close()
    
    print(f"Test report saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    # Example usage
    camera_params = VRCameraParameters()
    generator = AdvancedStereoGenerator(camera_params)
    
    # Generate a test scene
    left, right, ground_truth = generator.create_realistic_scene(
        scene_type="indoor",
        lighting="normal",
        add_noise=True
    )
    
    # Visualize
    vis = DisparityVisualization()
    colored_disp = vis.create_disparity_colormap(ground_truth.disparity_map)
    
    # Show results
    cv2.imshow('Left', left)
    cv2.imshow('Right', right)
    cv2.imshow('Disparity', colored_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()