module load cuda/cuda-12.1.0
module load anaconda
conda create --name splatter-image python=3.8
source activate splatter-image
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install rembg
pip install omegaconf
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization
cd diff-gaussian-rasterization
git submodule update --init --recursive
python setup.py build_ext --inplace
cd ..
python my_imageTo3D.py
