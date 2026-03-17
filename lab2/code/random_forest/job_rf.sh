#!/bin/bash

# sbatch random_forest/job_rf.sh

#SBATCH --job-name=lab2-part3
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_214

echo "=============================="
echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
nvidia-smi || true
echo "=============================="

echo "Step 1: extract AE latent vectors"
srun python extract_part3_latent_vectors.py \
  configs/finetune_final.yaml \
  checkpoints/finetune/final/final-epoch=004.ckpt \
  results/part3_latent_vectors.npz

echo "Step 2: train/evaluate random forest"
srun python random_forest/part3_random_forest.py \
  --ae-features results/part3_latent_vectors.npz \
  --labeled-paths \
    ../image_data_float32/O012791.npz \
    ../image_data_float32/O013257.npz \
    ../image_data_float32/O013490.npz \
  --outdir results/part3_random_forest

echo "=============================="
echo "Job finished on $(date)"
echo "=============================="