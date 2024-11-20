#!/bin/sh
#SBATCH --job-name=assignment_5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32
#SBATCH --mem=32G
##SBATCH --output=output{$JobID}.out
#SBATCH --error=errors-%J.out
#SBATCH --time=2:00:00

nvidia-smi
# python3 -m pip install --user -r requirements.txt
# python3 -m resize.py
# python3 -m pip install objaverse --upgrade
python3 -m main
