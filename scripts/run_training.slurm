#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --mem=16G
# Load required modules
module purge
module load gcccore/10.2.0 python/3.8.6

time srun python run_training_0.py