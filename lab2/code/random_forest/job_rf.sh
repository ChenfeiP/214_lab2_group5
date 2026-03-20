#!/bin/bash

# sbatch random_forest/job_rf.sh

#SBATCH --job-name=lab2-rf
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_214

cd "$CODE_DIR"
mkdir -p results/part3_random_forest

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
  transfer_learning/configs/finetune_final.yaml \
  results/transfer_learning/checkpoints_modified/finetune/final/final-epoch=004-v2.ckpt \
  results/part3_latent_vectors.npz

echo "Step 2: train/evaluate random forest"
srun python random_forest/part3_random_forest.py \
  --ae-features results/part3_latent_vectors.npz \
  --labeled-paths \
    ../data/O012791.npz \
    ../data/O013257.npz \
    ../data/O013490.npz \
  --outdir results/part3_random_forest \
  --random-state 42

echo "=============================="
echo "Job finished on $(date)"
echo "=============================="
