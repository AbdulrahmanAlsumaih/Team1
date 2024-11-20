# Point-E Model Runner and Visualizer

This repository contains scripts to run OpenAI's Point-E model for 3D point cloud generation and visualization.

## Installation

```bash
git clone https://github.com/openai/point-e.git
cd point-e
pip install -e .
```

## Files

### point_e_runner.py
Main script to generate 3D point clouds from text prompts using Point-E model. Initializes base and upsampler models, loads checkpoints, and generates point clouds.

### visualizer.py 
Visualization script that:
- Saves point cloud as PLY file for 3D software import
- Generates rotating MP4 animation of the point cloud

## Dependencies
- PyTorch
- Open3D
- FFmpeg
- Matplotlib

## Usage

1. Run the model:
```bash
python point_e_runner.py
```

2. Generate visualizations:
```bash
python visualizer.py
```

Outputs:
- output.ply: 3D point cloud file
- rotation.mp4: Rotating animation of the point cloud

## Parameters
- Adjust rotation speed, duration, and FPS in visualizer.py
- Modify prompt in point_e_runner.py to generate different objects