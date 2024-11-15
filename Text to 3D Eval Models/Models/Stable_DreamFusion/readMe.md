# Stable-Dreamfusion

A PyTorch implementation for text-to-3D model generation using Stable Diffusion, designed to create 3D content from text prompts.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ashawkey/stable-dreamfusion.git
   cd stable-dreamfusion
   ```

2. **(Optional) Create a Python Virtual Environment:**
   To avoid conflicts with other Python packages, it’s recommended to create a virtual environment.
   - Using `venv`:
     ```bash
     python -m venv venv_stable-dreamfusion
     source venv_stable-dreamfusion/bin/activate  # Activate in each terminal session
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Download Pre-Trained Models

To enable image-conditioned 3D generation, download the following pretrained models:

### Zero-1-to-3 Model for Diffusion Backend
```bash
cd pretrained/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```

### Omnidata Model for Depth and Normal Prediction
Ensure `gdown` is installed:
```bash
mkdir pretrained/omnidata
cd pretrained/omnidata
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t'  # omnidata_dpt_depth_v2.ckpt
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t'  # omnidata_dpt_normal_v2.ckpt
```

### DeepFloyd-IF Model
To use DeepFloyd-IF, log in to Hugging Face:
```bash
huggingface-cli login
```

### DMTet Model
Download pre-generated tetrahedron grids under `tets`. The 256-resolution grid can be found online.

## Build Extensions (Optional)

Extensions can be built at runtime or manually installed:
1. **Install all extensions automatically:**
   ```bash
   cd stable-dreamfusion
   bash scripts/install_ext.sh
   ```

2. **Install manually (example for raymarching):**
   ```bash
   pip install ./raymarching
   ```

## Taichi Backend (Optional)

To use Taichi for Instant-NGP (for performance without requiring CUDA), install with:
```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```

## Tested Environment
- **CUDA:** Version 11.6
- **Hardware:** NVIDIA V100 GPU

For more details, refer to the [original repository](https://github.com/ashawkey/stable-dreamfusion/tree/main).
