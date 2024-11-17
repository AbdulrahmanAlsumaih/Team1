Here's the updated README with additional information on the `setup.sh` script:

# README for 3D Video Generation App

## Overview

This project is a Python-based application that generates images from text prompts using Stable Diffusion and processes these images into 3D video renderings. It features an intuitive web-based interface powered by Gradio. The app consists of the following main components:

- **SDXLImageGenerator**: Generates images based on text prompts using the Stable Diffusion XL model.
- **Image3DProcessor**: Processes the generated image to create a 3D model and exports it as a video.
- **Gradio Interface**: Provides a user-friendly web interface for interaction.

## Features

- Generate high-quality images from user prompts.
- Process and convert these images into 3D videos.
- Display the generated image and 3D video through an interactive web interface.

## Prerequisites

Ensure you have Python 3.x installed, along with the following dependencies:

- `torch>=1.8.0`
- `torchvision>=0.9.0`
- `numpy`
- `Pillow`
- `imageio`
- `rembg`
- `omegaconf`
- `huggingface_hub`
- `diffusers`
- `transformers`
- `gradio`

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo/3d-video-generator.git
   cd 3d-video-generator
   ```

2. **Run the setup script**:
   Execute the provided `setup.sh` script to install necessary dependencies and clone required repositories:

   ```bash
   bash setup.sh
   ```

   This script:

   - Installs `libglm-dev` for necessary graphics libraries.
   - Clones the `splatter-image` repository if it does not already exist and installs its dependencies.
   - Clones the `diff-gaussian-rasterization` repository if it does not already exist and builds the required C extensions.

3. **Install the remaining Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure the `requirements.txt` file includes:

   ```text
   torch>=1.8.0
   torchvision>=0.9.0
   numpy
   Pillow
   imageio
   rembg
   omegaconf
   huggingface_hub
   diffusers
   transformers
   gradio
   ```

4. **Prepare model files**:
   - Place the configuration file (`gradio_config.yaml`) in the appropriate path (`/home/user/app/splatter-image/`).
   - Ensure the pre-trained model (`model_latest.pth`) is in the required path or downloaded via `hf_hub_download`.

## Usage

To run the application:

```bash
python app.py
```

### Workflow

1. Launch the Gradio app and open the provided URL in your web browser.
2. Enter a text prompt into the input field and click "Generate 3D object".
3. The generated image will appear on the left panel.
4. Once the image is ready, it will be processed into a 3D video, displayed on the right panel.

## Code Structure

### `setup.sh` Script

The `setup.sh` script performs the following:

- Installs `libglm-dev` for graphics support.
- Clones the `splatter-image` repository and installs its dependencies.
- Clones the `diff-gaussian-rasterization` repository and builds C extensions.

### SDXLImageGenerator Class

- Initializes a diffusion pipeline using the Stable Diffusion XL model.
- Generates images from user prompts and outputs them in PNG format.

### Image3DProcessor Class

- Loads and configures a 3D model using `GaussianSplatPredictor`.
- Preprocesses the image (optionally removes background and adjusts size).
- Reconstructs the 3D model and exports it as a video.

### GradioApp Class

- Integrates the image generation and 3D processing functions.
- Creates a Gradio interface with inputs and outputs connected to corresponding functions.

## Configuration Details

- **`model_cfg_path`**: Path to the configuration file (`gradio_config.yaml`).
- **`model_repo_id`**: Identifier for the Hugging Face model repository.
- **`model_filename`**: Name of the pre-trained model file.

## Notes

- Ensure your system has a compatible CUDA-capable GPU for optimal performance, as this app uses GPU acceleration if available.
- The video export is done using `imageio`, and the video output format is `.mp4`.

## License

Specify the appropriate license under which your code is distributed (e.g., MIT, Apache 2.0).

## Contact

For any issues, questions, or contributions, feel free to open an issue or submit a pull request on the repository.

---

This README provides all the essential details to understand, install, and run the 3D video generation application, including details on the `setup.sh` script and its functions.
