#!/usr/bin/env bash
# Part 3 — logistic regression only (after TL post job has written embeddings).
# Depends on: transfer_learning/job_post_tl.sh completed successfully.
#
#SBATCH --job-name=part3-post-lr
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-part3-post-lr-%j.out
#SBATCH --error=slurm-part3-post-lr-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_214

echo "=============================="
echo "[part3-post-lr] started $(date)"
echo "Host: $(hostname)"
echo "CODE_DIR: $CODE_DIR"
echo "Python: $(which python)"
python -V
echo "=============================="

echo "[INFO] Logistic regression (LOIO + comparison)"
srun python logistic_regression/logistic_experiments.py --variant modified
srun python logistic_regression/logistic_experiments.py --variant baseline
srun python logistic_regression/compare_logistic_results.py

echo "=============================="
echo "[part3-post-lr] finished $(date)"
echo "=============================="
