#!/usr/bin/env python3
"""
VR Video Quality Assessment Script using PyIQA

This script processes VR video files (fisheye or equirectangular projection) and
assesses their quality using multiple metrics from the pyiqa library.

Features:
- Splits VR frames in half for separate eye evaluation
- Handles fisheye and equirectangular projections
- Uses multiple quality assessment metrics
- Exports results to CSV
- Shows progress with tqdm
"""

import os
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import cv2
import numpy as np
import torch
import pyiqa
from tqdm import tqdm
import pandas as pd


class VRVideoQualityAssessor:
    """Main class for assessing VR video quality using multiple metrics."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the VR Video Quality Assessor.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Define the metrics we'll use - a robust collection covering different aspects
        self.metric_names = [
            # Full-reference metrics (if we have reference)
            'psnr',      # Peak Signal-to-Noise Ratio
            'ssim',      # Structural Similarity Index
            'ms_ssim',   # Multi-scale SSIM
            'vif',       # Visual Information Fidelity
            'fsim',      # Feature Similarity Index
            'gmsd',      # Gradient Magnitude Similarity Deviation
            'vsi',       # Visual Saliency Index
            'lpips',     # Learned Perceptual Image Patch Similarity
            
            # No-reference metrics (for standalone quality assessment)
            'brisque',   # Blind/Referenceless Image Spatial Quality Evaluator
            'niqe',      # Natural Image Quality Evaluator
            'musiq',     # Multi-scale Image Quality Transformer
            'dbcnn',     # Deep Bilinear CNN
            'nima',      # Neural Image Assessment
            'clipiqa',   # CLIP-based Image Quality Assessment
        ]
        
        # Initialize metrics
        self.metrics = {}
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all quality assessment metrics."""
        print(f"Initializing metrics on {self.device}...")
        
        for metric_name in self.metric_names:
            try:
                # Some metrics might not be available or might fail to initialize
                self.metrics[metric_name] = pyiqa.create_metric(metric_name, device=self.device)
                print(f"✓ Initialized {metric_name}")
            except Exception as e:
                print(f"✗ Failed to initialize {metric_name}: {e}")
                
    def process_frame(self, frame: np.ndarray, projection_type: str = 'equirectangular') -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a VR frame by splitting it for each eye and handling projection.
        
        Args:
            frame: Input frame (H, W, C)
            projection_type: 'equirectangular' or 'fisheye'
            
        Returns:
            Tuple of (left_eye_frame, right_eye_frame)
        """
        height, width = frame.shape[:2]
        
        # Split frame in half for left and right eye
        mid_point = width // 2
        left_eye = frame[:, :mid_point]
        right_eye = frame[:, mid_point:]
        
        # Handle different projection types
        if projection_type == 'fisheye':
            left_eye = self._process_fisheye(left_eye)
            right_eye = self._process_fisheye(right_eye)
        elif projection_type == 'equirectangular':
            left_eye = self._process_equirectangular(left_eye)
            right_eye = self._process_equirectangular(right_eye)
            
        return left_eye, right_eye
    
    def _process_fisheye(self, frame: np.ndarray) -> np.ndarray:
        """
        Process fisheye projection by cropping to remove black areas.
        
        Args:
            frame: Fisheye frame
            
        Returns:
            Cropped frame
        """
        # Convert to grayscale for black area detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find non-black pixels
        coords = cv2.findNonZero(gray > 10)
        
        if coords is not None:
            # Get bounding box of non-black region
            x, y, w, h = cv2.boundingRect(coords)
            
            # Crop to square (typical for fisheye)
            size = min(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            
            x_start = max(0, center_x - size // 2)
            y_start = max(0, center_y - size // 2)
            x_end = min(frame.shape[1], x_start + size)
            y_end = min(frame.shape[0], y_start + size)
            
            return frame[y_start:y_end, x_start:x_end]
        
        return frame
    
    def _process_equirectangular(self, frame: np.ndarray) -> np.ndarray:
        """
        Process equirectangular projection by cropping to central region.
        
        Args:
            frame: Equirectangular frame
            
        Returns:
            Cropped frame
        """
        height, width = frame.shape[:2]
        
        # Crop to central 80% to avoid distortion at poles
        crop_h = int(height * 0.1)
        crop_w = int(width * 0.1)
        
        return frame[crop_h:-crop_h, crop_w:-crop_w]
    
    def assess_frame_quality(self, left_eye: np.ndarray, right_eye: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Assess quality of left and right eye frames using all metrics.
        
        Args:
            left_eye: Left eye frame
            right_eye: Right eye frame
            
        Returns:
            Dictionary with quality scores for each eye and metric
        """
        results = {'left': {}, 'right': {}}
        
        # Convert frames to tensors
        left_tensor = self._prepare_tensor(left_eye)
        right_tensor = self._prepare_tensor(right_eye)
        
        # Evaluate each metric
        for metric_name, metric in self.metrics.items():
            try:
                # Check if metric is no-reference (single input) or full-reference
                if metric_name in ['brisque', 'niqe', 'musiq', 'dbcnn', 'nima', 'clipiqa']:
                    # No-reference metrics
                    results['left'][metric_name] = float(metric(left_tensor).cpu().item())
                    results['right'][metric_name] = float(metric(right_tensor).cpu().item())
                else:
                    # For full-reference metrics, we compare against a reference
                    # In this case, we'll use a slightly blurred version as reference
                    # In practice, you might want to use actual reference frames
                    ref_left = cv2.GaussianBlur(left_eye, (3, 3), 0)
                    ref_right = cv2.GaussianBlur(right_eye, (3, 3), 0)
                    ref_left_tensor = self._prepare_tensor(ref_left)
                    ref_right_tensor = self._prepare_tensor(ref_right)
                    
                    results['left'][metric_name] = float(metric(left_tensor, ref_left_tensor).cpu().item())
                    results['right'][metric_name] = float(metric(right_tensor, ref_right_tensor).cpu().item())
                    
            except Exception as e:
                # If metric fails, record as NaN
                results['left'][metric_name] = float('nan')
                results['right'][metric_name] = float('nan')
                warnings.warn(f"Failed to compute {metric_name}: {e}")
                
        return results
    
    def _prepare_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert numpy frame to torch tensor for metric computation.
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Torch tensor (1, C, H, W) normalized to [0, 1]
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def process_video(self, video_path: str, projection_type: str = 'equirectangular',
                     sample_interval: int = 30) -> List[Dict]:
        """
        Process a single video file and compute quality metrics.
        
        Args:
            video_path: Path to video file
            projection_type: 'equirectangular' or 'fisheye'
            sample_interval: Sample every N frames (default: 30)
            
        Returns:
            List of quality assessment results for each sampled frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        frame_count = 0
        
        pbar = tqdm(total=total_frames // sample_interval, 
                   desc=f"Processing {Path(video_path).name}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                # Process frame
                left_eye, right_eye = self.process_frame(frame, projection_type)
                
                # Assess quality
                quality_scores = self.assess_frame_quality(left_eye, right_eye)
                
                # Add metadata
                result = {
                    'video': Path(video_path).name,
                    'frame': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0
                }
                
                # Flatten the nested dictionary
                for eye in ['left', 'right']:
                    for metric, score in quality_scores[eye].items():
                        result[f'{eye}_{metric}'] = score
                
                results.append(result)
                pbar.update(1)
                
            frame_count += 1
            
        cap.release()
        pbar.close()
        
        return results
    
    def process_folder(self, folder_path: str, output_csv: str,
                      projection_type: str = 'equirectangular',
                      video_extensions: List[str] = None,
                      sample_interval: int = 30):
        """
        Process all video files in a folder.
        
        Args:
            folder_path: Path to folder containing videos
            output_csv: Path to output CSV file
            projection_type: 'equirectangular' or 'fisheye'
            video_extensions: List of video file extensions to process
            sample_interval: Sample every N frames
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(folder_path).glob(f'*{ext}'))
            video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
        if not video_files:
            raise ValueError(f"No video files found in {folder_path}")
            
        print(f"Found {len(video_files)} video files to process")
        
        all_results = []
        
        # Process each video
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                results = self.process_video(str(video_path), projection_type, sample_interval)
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
                
        # Save results to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print("-" * 50)
            
            # Get numeric columns (metric scores)
            metric_cols = [col for col in df.columns if any(metric in col for metric in self.metric_names)]
            
            for col in metric_cols:
                if df[col].notna().any():
                    print(f"{col}:")
                    print(f"  Mean: {df[col].mean():.4f}")
                    print(f"  Std:  {df[col].std():.4f}")
                    print(f"  Min:  {df[col].min():.4f}")
                    print(f"  Max:  {df[col].max():.4f}")
                    print()
        else:
            print("No results to save!")


def main():
    """Main function to run the VR video quality assessment."""
    parser = argparse.ArgumentParser(description='Assess quality of VR video files using PyIQA')
    parser.add_argument('input_folder', help='Path to folder containing VR video files')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument('--projection', choices=['equirectangular', 'fisheye'], 
                       default='equirectangular', help='VR projection type')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to run computations on')
    parser.add_argument('--sample-interval', type=int, default=30,
                       help='Sample every N frames (default: 30)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                       help='Video file extensions to process')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        
    # Create assessor
    assessor = VRVideoQualityAssessor(device=args.device)
    
    # Process videos
    assessor.process_folder(
        args.input_folder,
        args.output_csv,
        projection_type=args.projection,
        video_extensions=args.extensions,
        sample_interval=args.sample_interval
    )


if __name__ == '__main__':
    main()