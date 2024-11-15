#!/bin/bash
#SBATCH --job-name=model4
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.log
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sa181349@ucf.edu

module load anaconda
module load gcc/gcc-11.2.0
conda activate threestudio

export PATH=/home/cap5415.student23/cuda/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/cap5415.student23/cuda/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

srun python m3d.py
