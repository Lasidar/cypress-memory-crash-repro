#!/usr/bin/env python3
"""
Example usage of the VR Video Quality Assessment tool.

This script demonstrates various ways to use the VRVideoQualityAssessor class.
"""

import os
import numpy as np
from vr_video_quality_assessment import VRVideoQualityAssessor
import pandas as pd


def example_single_video():
    """Example: Process a single VR video file."""
    print("=" * 60)
    print("Example 1: Processing a single VR video")
    print("=" * 60)
    
    # Create assessor (will use GPU if available)
    assessor = VRVideoQualityAssessor()
    
    # Process a single video
    video_path = "sample_vr_video.mp4"  # Replace with your video path
    
    if os.path.exists(video_path):
        results = assessor.process_video(
            video_path,
            projection_type='equirectangular',
            sample_interval=60  # Sample every 60 frames (2 seconds at 30fps)
        )
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        
        print(f"\nProcessed {len(results)} frames from {video_path}")
        print("\nFirst 5 results:")
        print(df.head())
        
        # Calculate average quality scores
        print("\nAverage quality scores:")
        metric_columns = [col for col in df.columns if col.startswith(('left_', 'right_'))]
        for col in metric_columns:
            if df[col].notna().any():
                print(f"{col}: {df[col].mean():.4f}")
    else:
        print(f"Video file not found: {video_path}")


def example_batch_processing():
    """Example: Process multiple VR videos in a folder."""
    print("\n" + "=" * 60)
    print("Example 2: Batch processing VR videos")
    print("=" * 60)
    
    # Create assessor with custom device
    assessor = VRVideoQualityAssessor(device='cuda')  # Force GPU usage
    
    # Process all videos in a folder
    video_folder = "./vr_videos"  # Replace with your folder path
    output_csv = "vr_quality_results.csv"
    
    if os.path.exists(video_folder):
        assessor.process_folder(
            video_folder,
            output_csv,
            projection_type='fisheye',  # For fisheye VR videos
            video_extensions=['.mp4', '.mov'],  # Only process these formats
            sample_interval=30  # Sample every second at 30fps
        )
        
        # Load and analyze results
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            
            # Group by video and calculate statistics
            print("\nQuality summary by video:")
            video_groups = df.groupby('video')
            
            for video_name, group in video_groups:
                print(f"\n{video_name}:")
                # Calculate mean scores for each metric
                metric_cols = [col for col in group.columns if col.startswith(('left_', 'right_'))]
                for col in metric_cols[:5]:  # Show first 5 metrics
                    if group[col].notna().any():
                        print(f"  {col}: mean={group[col].mean():.3f}, std={group[col].std():.3f}")
    else:
        print(f"Video folder not found: {video_folder}")


def example_custom_metrics():
    """Example: Use only specific metrics for faster processing."""
    print("\n" + "=" * 60)
    print("Example 3: Custom metric selection")
    print("=" * 60)
    
    # Create assessor with custom metrics
    assessor = VRVideoQualityAssessor()
    
    # Override the default metrics list for faster processing
    assessor.metric_names = [
        'psnr',     # Fast traditional metric
        'ssim',     # Popular perceptual metric
        'brisque',  # No-reference metric
        'niqe'      # Natural image quality
    ]
    
    # Re-initialize with only selected metrics
    assessor.metrics = {}
    assessor._initialize_metrics()
    
    print(f"Using only {len(assessor.metrics)} metrics for faster processing")
    
    # Process video with custom metrics
    video_path = "sample_vr_video.mp4"
    if os.path.exists(video_path):
        results = assessor.process_video(video_path, sample_interval=120)
        print(f"Processed {len(results)} frames with custom metrics")


def example_quality_comparison():
    """Example: Compare quality between different VR videos or encodings."""
    print("\n" + "=" * 60)
    print("Example 4: Quality comparison between videos")
    print("=" * 60)
    
    assessor = VRVideoQualityAssessor()
    
    # Compare two different encodings of the same content
    video1 = "vr_video_high_quality.mp4"
    video2 = "vr_video_low_quality.mp4"
    
    results_comparison = {}
    
    for video_path in [video1, video2]:
        if os.path.exists(video_path):
            results = assessor.process_video(video_path, sample_interval=60)
            df = pd.DataFrame(results)
            
            # Calculate average scores
            avg_scores = {}
            metric_cols = [col for col in df.columns if col.startswith(('left_', 'right_'))]
            for col in metric_cols:
                if df[col].notna().any():
                    avg_scores[col] = df[col].mean()
            
            results_comparison[video_path] = avg_scores
    
    # Compare results
    if len(results_comparison) == 2:
        print("\nQuality comparison:")
        print("-" * 40)
        
        metrics = list(next(iter(results_comparison.values())).keys())
        for metric in metrics[:5]:  # Show first 5 metrics
            print(f"\n{metric}:")
            for video, scores in results_comparison.items():
                if metric in scores:
                    print(f"  {os.path.basename(video)}: {scores[metric]:.4f}")


def example_temporal_analysis():
    """Example: Analyze quality changes over time in a video."""
    print("\n" + "=" * 60)
    print("Example 5: Temporal quality analysis")
    print("=" * 60)
    
    assessor = VRVideoQualityAssessor()
    
    video_path = "sample_vr_video.mp4"
    if os.path.exists(video_path):
        # Process with frequent sampling for temporal analysis
        results = assessor.process_video(video_path, sample_interval=15)  # Every 0.5 seconds
        df = pd.DataFrame(results)
        
        # Analyze quality changes over time
        print("\nTemporal quality analysis:")
        
        # Select a few metrics to analyze
        metrics_to_analyze = ['left_ssim', 'right_ssim', 'left_brisque', 'right_brisque']
        
        for metric in metrics_to_analyze:
            if metric in df.columns and df[metric].notna().any():
                values = df[metric].values
                timestamps = df['timestamp'].values
                
                # Find quality drops (significant changes)
                if len(values) > 1:
                    changes = np.abs(np.diff(values))
                    threshold = np.std(changes) * 2  # 2 standard deviations
                    
                    significant_changes = np.where(changes > threshold)[0]
                    
                    print(f"\n{metric}:")
                    print(f"  Average: {np.mean(values):.4f}")
                    print(f"  Std Dev: {np.std(values):.4f}")
                    
                    if len(significant_changes) > 0:
                        print(f"  Significant changes at timestamps:")
                        for idx in significant_changes[:3]:  # Show first 3
                            print(f"    {timestamps[idx]:.1f}s: {values[idx]:.4f} -> {values[idx+1]:.4f}")


if __name__ == "__main__":
    print("VR Video Quality Assessment - Usage Examples")
    print("=" * 60)
    
    # Run examples (comment out those you don't need)
    
    # example_single_video()
    # example_batch_processing()
    example_custom_metrics()
    # example_quality_comparison()
    # example_temporal_analysis()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)