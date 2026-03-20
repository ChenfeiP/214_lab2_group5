#!/bin/bash

# From lab2/code:
#   sbatch job.sh transfer_learning/configs/pretrain.yaml
#   sbatch job.sh transfer_learning/configs/finetune_cv.yaml
#   sbatch job.sh transfer_learning/configs/finetune_final.yaml
#
# Baseline: use transfer_learning/configs/pretrain_baseline.yaml (etc.)

#SBATCH --job-name=lab2-autoencoder
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

if [ -z "$1" ]; then
    echo "Usage: sbatch job.sh <config_path>"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_214

echo "=============================="
echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: $1"
echo "Python: $(which python)"
python --version
echo "=============================="

srun python run_autoencoder.py "$1"

echo "=============================="
echo "Job finished on $(date)"
echo "=============================="
