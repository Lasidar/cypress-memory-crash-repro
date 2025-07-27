# VR Video Quality Assessment Tool

A Python script for assessing the quality of VR (Virtual Reality) video files using the PyIQA library. This tool is specifically designed to handle VR video formats with side-by-side stereo views and supports both equirectangular and fisheye projections.

## Features

- **Stereo VR Support**: Automatically splits VR frames in half to evaluate left and right eye views separately
- **Projection Handling**: 
  - **Equirectangular**: Crops central region to avoid pole distortions
  - **Fisheye**: Automatically detects and crops black areas around the circular image
- **Comprehensive Metrics**: Uses 14 different quality assessment metrics from PyIQA:
  - Full-reference metrics: PSNR, SSIM, MS-SSIM, VIF, FSIM, GMSD, VSI, LPIPS
  - No-reference metrics: BRISQUE, NIQE, MUSIQ, DBCNN, NIMA, CLIPIQA
- **Batch Processing**: Process entire folders of VR videos with progress tracking
- **CSV Export**: Results are exported to CSV format for easy analysis
- **GPU Acceleration**: Supports CUDA for faster processing

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd vr-video-quality-assessment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU acceleration, ensure you have CUDA-compatible PyTorch installed:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

Process a folder of VR videos:
```bash
python vr_video_quality_assessment.py /path/to/videos output_results.csv
```

### Advanced Options

```bash
python vr_video_quality_assessment.py /path/to/videos output_results.csv \
    --projection fisheye \
    --device cuda \
    --sample-interval 60 \
    --extensions .mp4 .mov
```

### Command Line Arguments

- `input_folder`: Path to folder containing VR video files
- `output_csv`: Path where the results CSV will be saved
- `--projection`: VR projection type (`equirectangular` or `fisheye`, default: `equirectangular`)
- `--device`: Computing device (`cuda` or `cpu`, default: `cuda` if available)
- `--sample-interval`: Sample every N frames (default: 30)
- `--extensions`: Video file extensions to process (default: `.mp4 .avi .mov .mkv .webm`)

## Output Format

The script generates a CSV file with the following columns:
- `video`: Video filename
- `frame`: Frame number
- `timestamp`: Timestamp in seconds
- `left_<metric>`: Quality score for left eye view
- `right_<metric>`: Quality score for right eye view

Example output:
```csv
video,frame,timestamp,left_psnr,right_psnr,left_ssim,right_ssim,...
video1.mp4,0,0.0,28.5,28.3,0.92,0.91,...
video1.mp4,30,1.0,27.8,27.9,0.90,0.89,...
```

## Metrics Explained

### Full-Reference Metrics (require reference for comparison)
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **MS-SSIM**: Multi-scale SSIM (0-1, higher is better)
- **VIF**: Visual Information Fidelity (higher is better)
- **FSIM**: Feature Similarity Index (0-1, higher is better)
- **GMSD**: Gradient Magnitude Similarity Deviation (lower is better)
- **VSI**: Visual Saliency Index (0-1, higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

### No-Reference Metrics (standalone quality assessment)
- **BRISQUE**: Blind/Referenceless Image Spatial Quality Evaluator (lower is better)
- **NIQE**: Natural Image Quality Evaluator (lower is better)
- **MUSIQ**: Multi-scale Image Quality Transformer (higher is better)
- **DBCNN**: Deep Bilinear CNN (higher is better)
- **NIMA**: Neural Image Assessment (1-10, higher is better)
- **CLIPIQA**: CLIP-based Image Quality Assessment (0-1, higher is better)

## Example Python Usage

```python
from vr_video_quality_assessment import VRVideoQualityAssessor

# Create assessor
assessor = VRVideoQualityAssessor(device='cuda')

# Process a single video
results = assessor.process_video('path/to/video.mp4', projection_type='equirectangular')

# Process a folder
assessor.process_folder('path/to/videos', 'results.csv', projection_type='fisheye')
```

## Performance Tips

1. **GPU Usage**: Use `--device cuda` for 5-10x faster processing
2. **Sample Interval**: Increase `--sample-interval` to process fewer frames for faster results
3. **Metric Selection**: Modify the `metric_names` list in the code to use only specific metrics
4. **Batch Size**: For very high resolution videos, you may need to reduce batch processing

## Troubleshooting

1. **CUDA Out of Memory**: Try processing on CPU or reduce the number of metrics
2. **Video Not Opening**: Ensure you have the proper codecs installed (ffmpeg)
3. **Slow Processing**: Use GPU acceleration and increase sample interval
4. **Missing Metrics**: Some metrics may require additional dependencies or models

## Requirements

- Python 3.7+
- PyTorch 1.9+
- OpenCV 4.5+
- 4GB+ GPU memory recommended for CUDA acceleration

## Citation

If you use this tool in your research, please cite the PyIQA library:

```bibtex
@misc{pyiqa,
  title={PyIQA: PyTorch Image Quality Assessment},
  author={Chaofeng Chen and Jiadi Mo},
  year={2022},
  url={https://github.com/chaofengc/IQA-PyTorch}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

