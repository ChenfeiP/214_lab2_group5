#!/usr/bin/env bash
set -euo pipefail

# ========= paths =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR"
LAB2_DIR="$(cd "$CODE_DIR/.." && pwd)"
ENV_YAML="$CODE_DIR/environment.yaml"
ENV_NAME="env_214"
DATA_DIR="$LAB2_DIR/data"
FLOAT32_DIR="$LAB2_DIR/data"
RESULTS_DIR="$CODE_DIR/results"

echo "[INFO] CODE_DIR    = $CODE_DIR"
echo "[INFO] LAB2_DIR    = $LAB2_DIR"
echo "[INFO] ENV_YAML    = $ENV_YAML"
echo "[INFO] DATA_DIR    = $DATA_DIR"
echo "[INFO] FLOAT32_DIR = $FLOAT32_DIR"
echo "[INFO] RESULTS_DIR = $RESULTS_DIR"

cd "$CODE_DIR"

# ========= conda init =========
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[INFO] Updating conda environment from $ENV_YAML ..."
conda env update -n "$ENV_NAME" -f "$ENV_YAML" --prune

echo "[INFO] Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "[INFO] Python executable: $(which python)"
python -V

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/part3_random_forest"
mkdir -p "$RESULTS_DIR/transfer_learning/comparisons"
mkdir -p "$RESULTS_DIR/transfer_learning/results_baseline"
mkdir -p "$RESULTS_DIR/transfer_learning/results_modified"
mkdir -p "$RESULTS_DIR/part3_logistic_regression/results_baseline"
mkdir -p "$RESULTS_DIR/part3_logistic_regression/results_modified"
mkdir -p "$FLOAT32_DIR"

# ========= convert npz float64 -> float32 =========
echo "[INFO] Running float32 conversion from $DATA_DIR to $FLOAT32_DIR ..."
python - <<PY
import numpy as np
from pathlib import Path

src_dir = Path("$DATA_DIR")
dst_dir = Path("$FLOAT32_DIR")
dst_dir.mkdir(parents=True, exist_ok=True)

npz_files = sorted(src_dir.glob("*.npz"))

for f in npz_files:
    with np.load(f) as data:
        save_dict = {}
        for key in data.files:
            arr = data[key]
            if np.issubdtype(arr.dtype, np.floating):
                save_dict[key] = arr.astype(np.float32)
            else:
                save_dict[key] = arr
    out_path = dst_dir / f.name
    np.savez_compressed(out_path, **save_dict)
    print(f"[INFO] saved {out_path}")
PY

# ========= EDA =========
echo "[INFO] Running EDA..."
python eda.py

# ========= Transfer Learning (baseline — parallel second chain for compare_transfer_results) =========
echo "[INFO] Submitting transfer learning jobs (baseline)..."

PRETRAIN_BASE_JOBID=$(sbatch job.sh transfer_learning/configs/pretrain_baseline.yaml | awk '{print $4}')
echo "[INFO] pretrain (baseline) job id: $PRETRAIN_BASE_JOBID"

FINETUNE_CV_BASE_JOBID=$(sbatch --dependency=afterok:"$PRETRAIN_BASE_JOBID" job.sh transfer_learning/configs/finetune_cv_baseline.yaml | awk '{print $4}')
echo "[INFO] finetune_cv (baseline) job id: $FINETUNE_CV_BASE_JOBID"

FINETUNE_FINAL_BASE_JOBID=$(sbatch --dependency=afterok:"$FINETUNE_CV_BASE_JOBID" job.sh transfer_learning/configs/finetune_final_baseline.yaml | awk '{print $4}')
echo "[INFO] finetune_final (baseline) job id: $FINETUNE_FINAL_BASE_JOBID"

# ========= Transfer Learning (modified — same chain as before, configs under transfer_learning/configs) =========
echo "[INFO] Submitting transfer learning jobs (modified)..."

PRETRAIN_MOD_JOBID=$(sbatch job.sh transfer_learning/configs/pretrain.yaml | awk '{print $4}')
echo "[INFO] pretrain (modified) job id: $PRETRAIN_MOD_JOBID"

FINETUNE_CV_MOD_JOBID=$(sbatch --dependency=afterok:"$PRETRAIN_MOD_JOBID" job.sh transfer_learning/configs/finetune_cv.yaml | awk '{print $4}')
echo "[INFO] finetune_cv (modified) job id: $FINETUNE_CV_MOD_JOBID"

FINETUNE_FINAL_MOD_JOBID=$(sbatch --dependency=afterok:"$FINETUNE_CV_MOD_JOBID" job.sh transfer_learning/configs/finetune_final.yaml | awk '{print $4}')
echo "[INFO] finetune_final (modified) job id: $FINETUNE_FINAL_MOD_JOBID"

# ========= Post TL (embeddings + TL analysis; after both finetune_final jobs) =========
echo "[INFO] Submitting Part 3 post-TL job (get_embedding, probes, viz, compare_transfer)..."
PART3_POST_TL_JOBID=$(
    sbatch --dependency=afterok:"$FINETUNE_FINAL_MOD_JOBID:$FINETUNE_FINAL_BASE_JOBID" \
        "$CODE_DIR/transfer_learning/job_post_tl.sh" | awk '{print $4}'
)
echo "[INFO] part3 post-TL job id: $PART3_POST_TL_JOBID"

# ========= Model A : Random Forest (unchanged: after modified finetune_final only) =========
echo "[INFO] Submitting random forest jobs..."
RF_JOBID=$(
    sbatch --dependency=afterok:"$FINETUNE_FINAL_MOD_JOBID" random_forest/job_rf.sh | awk '{print $4}'
)
echo "[INFO] random forest job id: $RF_JOBID"

# ========= Model B : Logistic Regression (after post-TL embeddings exist) =========
echo "[INFO] Submitting Part 3 post-LR job (logistic_experiments + compare)..."
PART3_POST_LR_JOBID=$(
    sbatch --dependency=afterok:"$PART3_POST_TL_JOBID" \
        "$CODE_DIR/logistic_regression/job_lr.sh" | awk '{print $4}'
)
echo "[INFO] part3 post-LR job id: $PART3_POST_LR_JOBID"

echo "[INFO] Deactivating environment: $ENV_NAME"
conda deactivate
