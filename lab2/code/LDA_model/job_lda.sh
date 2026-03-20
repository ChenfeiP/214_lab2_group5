#!/bin/bash

# sbatch LDA_model/job_lda.sh

#SBATCH --job-name=lab2-lda
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LDA_DIR="$SCRIPT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_214

cd "$CODE_DIR"
mkdir -p "$LDA_DIR/results/part3_lda"

echo "=============================="
echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
nvidia-smi || true
echo "=============================="

echo "Step 1: extract AE latent vectors for LDA"
srun python extract_part3_latent_vectors.py \
  transfer_learning/configs/finetune_final.yaml \
  results/transfer_learning/checkpoints_modified/finetune/final/final-epoch=004.ckpt \
  LDA_model/results/part3_lda/part3_latent_vectors.npz

echo "Step 2: train/evaluate LDA"
srun python LDA_model/LDA_model.py \
  --ae-features LDA_model/results/part3_lda/part3_latent_vectors.npz \
  --labeled-paths \
    ../data/O012791.npz \
    ../data/O013257.npz \
    ../data/O013490.npz \
  --outdir LDA_model/results/part3_lda

echo "=============================="
echo "Job finished on $(date)"
echo "=============================="
