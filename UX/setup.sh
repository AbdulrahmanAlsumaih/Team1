#!/bin/bash

# Update package lists and install libglm-dev
apt-get update && apt-get install -y libglm-dev

# Clone the repositories if they do not already exist
if [ ! -d "./splatter-image" ]; then
    echo "Cloning splatter-image repository..."
    git clone https://github.com/szymanowiczs/splatter-image.git
    cd splatter-image
    pip install -r requirements.txt
    pip install rembg
    pip install omegaconf
    cd ..
else
    echo "splatter-image repository already exists."
fi

if [ ! -d "./diff-gaussian-rasterization" ]; then
    echo "Cloning diff-gaussian-rasterization repository..."
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization
    cd diff-gaussian-rasterization
    python setup.py build_ext --inplace
    cd ..
else
    echo "diff-gaussian-rasterization repository already exists."
fi