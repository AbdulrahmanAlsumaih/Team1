# README for Group Project: 3D Image Reconstruction and Rendering

## Overview

This project involves reconstructing a 3D model from 2D images and rendering animations based on the reconstructed data. It utilizes several libraries including PyTorch for deep learning, OpenCV for image processing, and specific custom modules for Gaussian splatting techniques.

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.x
- PyTorch
- torchvision
- rembg
- omegaconf
- imageio

### Install Requirements

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
pip install rembg
pip install omegaconf
```

## Setup

1. **Clone the Repository**:
   Clone the repository containing the project files:

   ```bash
   git clone https://github.com/szymanowiczs/splatter-image.git
   cd splatter-image
   ```

2. **Clone Additional Dependencies**:
   This project requires additional repositories for Gaussian rasterization:

   ```bash
   git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization
   cd diff-gaussian-rasterization
   ```

3. **Install System Dependencies**:
   Install any system-level dependencies required by the Gaussian rasterization:

   ```bash
   apt-get install -y libglm-dev
   ```

4. **Build Extensions**:
   Build any necessary C++ extensions:

   ```bash
   python setup.py build_ext --inplace
   ```

## Usage

1. **Load Model Configuration**:
   Specify the path to the model configuration YAML file and load the model weights:

   ```python
   config_path = "path/to/gradio_config.yaml"
   model_cfg = OmegaConf.load(config_path)

   model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1",
                                filename="model_latest.pth")
   model = GaussianSplatPredictor(model_cfg)
   ckpt_loaded = torch.load(model_path, map_location="cuda")
   model.load_state_dict(ckpt_loaded["model_state_dict"])
   model.to("cuda")
   ```

2. **Preprocess Images**:
   Preprocess the input images by removing the background, resizing, and preparing them for reconstruction:

   ```python
   image = Image.open("path/to/image.jpg")
   processed_image = preprocess(image, preprocess_background=True, foreground_ratio=0.65)
   ```

3. **Reconstruct and Export**:
   Call the `reconstruct_and_export` function with the processed image to obtain the 3D model and a rendered video:

   ```python
   ply_out_path, loop_out_path = reconstruct_and_export(np.array(processed_image), "path/to/image.jpg")
   ```

   The paths to the exported files will be printed in the console.

## Example

An example workflow is provided at the end of the script, where a specific image is processed, reconstructed, and saved:

```python
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
image = Image.open(image_paths[0])
processed_image = preprocess(image, preprocess_background=True, foreground_ratio=0.65)
ply_out_path, loop_out_path = reconstruct_and_export(np.array(processed_image), image_paths[0])
```

## Outputs

- **3D Model**: Exported as a `.ply` file.
- **Rendered Video**: Exported as a `.mp4` file showing the animation of the reconstructed model.

## Notes

- Ensure you have a compatible GPU for running the model, as it utilizes CUDA for faster processing.
- Modify the paths in the example to point to your own images and configuration files.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
