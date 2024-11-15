
# Newton Set-up Guide

1. **Load GCC Module**
   ```bash
   module load gcc/gcc-11.2.0
   ```

2. **Clone Bits and Bytes Repository**
   ```bash
   git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
   ```

3. **Run CUDA Installation Script**
   Change directory to `bitsandbytes` if needed, and install the required CUDA version:
   ```bash
   cd bitsandbytes
   python install_cuda.py 118
   ```

4. **Set CUDA Paths**
   Update your environment paths to ensure proper CUDA configuration:
   ```bash
   export PATH=/home/cap6411.student2/cuda/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/home/cap6411.student2/cuda/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

5. **Refresh Shell Configuration**
   Load the new configurations by refreshing `.bashrc`:
   ```bash
   source ~/.bashrc
   ```

6. **Verify CUDA Installation**
   - Check `nvcc` version:
     ```bash
     nvcc --version
     ```
   - Verify NVIDIA drivers with `nvidia-smi`:
     ```bash
     nvidia-smi
     ```

This setup guide ensures you load the necessary modules, install CUDA, and update your paths, enabling proper CUDA functionality for your project.
