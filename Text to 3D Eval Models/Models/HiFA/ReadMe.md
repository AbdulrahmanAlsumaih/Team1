# threestudio Setup Guide

This repository provides the setup instructions for `threestudio`, designed for users with an NVIDIA graphics card with at least 6GB VRAM and CUDA installed.

## Installation

### Step 1: Clone the Repository
Start by cloning the `threestudio` repository and navigating to the project directory:
```bash
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio/
```

### Step 2: Set Up Python Environment
Ensure you have Python version 3.8 or higher installed. For better package management, it’s recommended to create a virtual environment using `conda`:

```bash
# Create the virtual environment
conda create --name virtualenv

# Activate the virtual environment
conda activate virtualenv
```

#### Update pip
For optimal performance, update pip to the latest version:
```bash
pip install --upgrade pip
```

### Step 3: Install PyTorch
Install PyTorch (version 1.12.1+cu113 or 2.0.0+cu118 recommended):

- **torch1.12.1+cu113**
  ```bash
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

- **torch2.0.0+cu118**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

### Step 4: (Optional) Install `ninja`
To speed up CUDA extension compilation, you can install `ninja`:
```bash
pip install ninja
```

### Step 5: Install Dependencies
Install the necessary packages by running:
```bash
pip install -r requirements.txt
```

> **Note**: Installing `tiny-cuda-nn` may require downgrading pip to version 23.0.1.

### Step 6: Running the Model
Once setup is complete, run the `_runner.py` file to start the model.

```bash
python _runner.py
```

## Original Repository
For further details, you can refer to the [original threestudio repository](https://github.com/threestudio-project/threestudio).
```
