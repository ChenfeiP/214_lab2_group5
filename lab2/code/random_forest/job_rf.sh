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

echo "Step 1: extract AE latent vectors for labeled data"
srun python extract_part3_latent_vectors.py \
  configs/finetune_final.yaml \
  results/transfer_learning/modified/finetune/final/final-epoch=004-v2.ckpt \
  results/part3_latent_vectors.npz

echo "Step 2: train/evaluate random forest"
srun python random_forest/part3_random_forest.py \
  --ae-features results/part3_latent_vectors.npz \
  --labeled-paths \
    ../data/O012791.npz \
    ../data/O013257.npz \
    ../data/O013490.npz \
  --outdir results/part3_random_forest \
  --random-state 42 \

echo "=============================="
echo "Job finished on $(date)"
echo "=============================="