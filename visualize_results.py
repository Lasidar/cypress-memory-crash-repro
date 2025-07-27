#!/usr/bin/env python3
"""
Visualize and analyze VR video quality assessment results.

This script creates plots and summaries from the CSV output of the VR quality assessment.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(csv_path):
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def plot_temporal_quality(df, output_dir):
    """Plot quality metrics over time for each video."""
    videos = df['video'].unique()
    
    # Select a subset of metrics to plot
    metrics_to_plot = [
        'left_ssim', 'right_ssim',
        'left_psnr', 'right_psnr',
        'left_brisque', 'right_brisque'
    ]
    
    # Filter to existing metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
    
    for video in videos:
        video_df = df[df['video'] == video].sort_values('timestamp')
        
        if len(metrics_to_plot) > 0:
            fig, axes = plt.subplots(len(metrics_to_plot), 1, 
                                   figsize=(12, 3*len(metrics_to_plot)), 
                                   sharex=True)
            
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            fig.suptitle(f'Quality Metrics Over Time - {video}', fontsize=16)
            
            for idx, metric in enumerate(metrics_to_plot):
                if video_df[metric].notna().any():
                    axes[idx].plot(video_df['timestamp'], video_df[metric], 
                                 marker='o', markersize=4, linewidth=1.5)
                    axes[idx].set_ylabel(metric)
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = video_df[metric].mean()
                    axes[idx].axhline(y=mean_val, color='r', linestyle='--', 
                                     alpha=0.5, label=f'Mean: {mean_val:.3f}')
                    axes[idx].legend()
            
            axes[-1].set_xlabel('Time (seconds)')
            
            plt.tight_layout()
            output_path = output_dir / f'temporal_{Path(video).stem}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved temporal plot: {output_path}")


def plot_eye_comparison(df, output_dir):
    """Compare quality between left and right eyes."""
    # Find pairs of left/right metrics
    metric_pairs = []
    for col in df.columns:
        if col.startswith('left_'):
            right_col = col.replace('left_', 'right_')
            if right_col in df.columns:
                metric_pairs.append((col, right_col))
    
    if not metric_pairs:
        print("No left/right metric pairs found")
        return
    
    # Create comparison plots
    n_pairs = len(metric_pairs)
    fig, axes = plt.subplots(1, min(n_pairs, 4), figsize=(4*min(n_pairs, 4), 4))
    
    if n_pairs == 1:
        axes = [axes]
    
    fig.suptitle('Left vs Right Eye Quality Comparison', fontsize=16)
    
    for idx, (left_metric, right_metric) in enumerate(metric_pairs[:4]):
        if df[left_metric].notna().any() and df[right_metric].notna().any():
            axes[idx].scatter(df[left_metric], df[right_metric], alpha=0.5, s=20)
            
            # Add diagonal line
            min_val = min(df[left_metric].min(), df[right_metric].min())
            max_val = max(df[left_metric].max(), df[right_metric].max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 
                         'r--', alpha=0.5, label='Equal quality')
            
            axes[idx].set_xlabel(f'Left Eye')
            axes[idx].set_ylabel(f'Right Eye')
            axes[idx].set_title(left_metric.replace('left_', ''))
            axes[idx].grid(True, alpha=0.3)
            
            # Calculate correlation
            corr = df[[left_metric, right_metric]].corr().iloc[0, 1]
            axes[idx].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                         transform=axes[idx].transAxes, 
                         verticalalignment='top')
    
    plt.tight_layout()
    output_path = output_dir / 'eye_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eye comparison plot: {output_path}")


def plot_metric_distributions(df, output_dir):
    """Plot distributions of quality metrics."""
    # Get all metric columns
    metric_cols = [col for col in df.columns 
                  if col.startswith(('left_', 'right_')) and df[col].notna().any()]
    
    if not metric_cols:
        print("No metric columns found")
        return
    
    # Create distribution plots
    n_metrics = len(metric_cols)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else axes
    
    fig.suptitle('Quality Metric Distributions', fontsize=16)
    
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx] if n_metrics > 1 else axes
        
        # Plot histogram and KDE
        df[metric].hist(bins=30, alpha=0.7, ax=ax, density=True)
        df[metric].plot.density(ax=ax, color='red', linewidth=2)
        
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        ax.axvline(mean_val, color='green', linestyle='--', 
                  label=f'Mean: {mean_val:.3f}')
        ax.text(0.95, 0.95, f'Std: {std_val:.3f}', 
               transform=ax.transAxes, 
               horizontalalignment='right',
               verticalalignment='top')
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'metric_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot: {output_path}")


def create_summary_report(df, output_dir):
    """Create a summary report of the quality assessment."""
    report_path = output_dir / 'quality_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("VR Video Quality Assessment Summary\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics
        f.write("Dataset Overview:\n")
        f.write(f"- Total videos: {df['video'].nunique()}\n")
        f.write(f"- Total frames analyzed: {len(df)}\n")
        f.write(f"- Time range: {df['timestamp'].min():.1f}s - {df['timestamp'].max():.1f}s\n")
        f.write("\n")
        
        # Per-video summary
        f.write("Per-Video Summary:\n")
        f.write("-" * 60 + "\n")
        
        for video in df['video'].unique():
            video_df = df[df['video'] == video]
            f.write(f"\n{video}:\n")
            f.write(f"  Frames analyzed: {len(video_df)}\n")
            f.write(f"  Duration: {video_df['timestamp'].max():.1f}s\n")
            
            # Get metric columns
            metric_cols = [col for col in video_df.columns 
                         if col.startswith(('left_', 'right_')) and video_df[col].notna().any()]
            
            if metric_cols:
                f.write("  Average scores:\n")
                for metric in sorted(metric_cols):
                    mean_score = video_df[metric].mean()
                    std_score = video_df[metric].std()
                    f.write(f"    {metric}: {mean_score:.4f} (±{std_score:.4f})\n")
        
        # Overall metric summary
        f.write("\n" + "=" * 60 + "\n")
        f.write("Overall Metric Statistics:\n")
        f.write("-" * 60 + "\n")
        
        metric_cols = [col for col in df.columns 
                      if col.startswith(('left_', 'right_')) and df[col].notna().any()]
        
        for metric in sorted(metric_cols):
            f.write(f"\n{metric}:\n")
            f.write(f"  Mean:   {df[metric].mean():.4f}\n")
            f.write(f"  Std:    {df[metric].std():.4f}\n")
            f.write(f"  Min:    {df[metric].min():.4f}\n")
            f.write(f"  Max:    {df[metric].max():.4f}\n")
            f.write(f"  Median: {df[metric].median():.4f}\n")
        
        # Eye difference analysis
        f.write("\n" + "=" * 60 + "\n")
        f.write("Left vs Right Eye Analysis:\n")
        f.write("-" * 60 + "\n")
        
        for col in df.columns:
            if col.startswith('left_'):
                right_col = col.replace('left_', 'right_')
                if right_col in df.columns:
                    metric_name = col.replace('left_', '')
                    left_mean = df[col].mean()
                    right_mean = df[right_col].mean()
                    diff = abs(left_mean - right_mean)
                    percent_diff = (diff / ((left_mean + right_mean) / 2)) * 100
                    
                    f.write(f"\n{metric_name}:\n")
                    f.write(f"  Left eye mean:  {left_mean:.4f}\n")
                    f.write(f"  Right eye mean: {right_mean:.4f}\n")
                    f.write(f"  Difference:     {diff:.4f} ({percent_diff:.1f}%)\n")
    
    print(f"Saved summary report: {report_path}")


def main():
    """Main function to run visualization and analysis."""
    parser = argparse.ArgumentParser(description='Visualize VR video quality assessment results')
    parser.add_argument('csv_file', help='Path to the results CSV file')
    parser.add_argument('--output-dir', default='quality_analysis', 
                       help='Directory to save plots and reports')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    df = load_results(args.csv_file)
    
    # Create summary report
    create_summary_report(df, output_dir)
    
    if not args.no_plots:
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Generate plots
        print("\nGenerating plots...")
        plot_temporal_quality(df, output_dir)
        plot_eye_comparison(df, output_dir)
        plot_metric_distributions(df, output_dir)
        
        print(f"\nAll outputs saved to: {output_dir}")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()