#!/usr/bin/env bash
# Part 3 — post transfer learning only: export embeddings + TL analysis.
# Depends on: both finetune_final (modified + baseline) succeeded.
# Submitted from run.sh; downstream logistic job depends on this job.
#
#SBATCH --job-name=part3-post-tl
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-part3-post-tl-%j.out
#SBATCH --error=slurm-part3-post-tl-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_214

pick_latest_ckpt() {
    local d="$1"
    if [[ ! -d "$d" ]]; then
        echo "[ERROR] checkpoint directory missing: $d" >&2
        return 1
    fi
    local f
    f="$(ls -t "$d"/*.ckpt 2>/dev/null | head -1 || true)"
    if [[ -z "$f" ]]; then
        echo "[ERROR] no .ckpt files in $d" >&2
        return 1
    fi
    printf '%s' "$f"
}

echo "=============================="
echo "[part3-post-tl] started $(date)"
echo "Host: $(hostname)"
echo "CODE_DIR: $CODE_DIR"
echo "Python: $(which python)"
python -V
echo "=============================="

CKPT_MOD="$(pick_latest_ckpt "$CODE_DIR/results/transfer_learning/checkpoints_modified/finetune/final")"
CKPT_BASE="$(pick_latest_ckpt "$CODE_DIR/results/transfer_learning/checkpoints_baseline/finetune/final")"
echo "[INFO] CKPT_MOD  = $CKPT_MOD"
echo "[INFO] CKPT_BASE = $CKPT_BASE"

echo "[INFO] Export embeddings (modified)"
srun python transfer_learning/get_embedding.py \
    transfer_learning/configs/finetune_final.yaml \
    "$CKPT_MOD"

echo "[INFO] Export embeddings (baseline)"
srun python transfer_learning/get_embedding.py \
    transfer_learning/configs/finetune_final_baseline.yaml \
    "$CKPT_BASE"

echo "[INFO] Transfer learning analysis"
srun python transfer_learning/quick_probe.py --variant modified
srun python transfer_learning/quick_probe.py --variant baseline
srun python transfer_learning/latent_dim_table.py --variant modified
srun python transfer_learning/latent_dim_table.py --variant baseline
srun python transfer_learning/visualize_embeddings.py --variant modified
srun python transfer_learning/visualize_embeddings.py --variant baseline
srun python transfer_learning/compare_transfer_results.py

echo "=============================="
echo "[part3-post-tl] finished $(date)"
echo "=============================="
