#!/usr/bin/env bash
set -euo pipefail

# ========= paths =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR"
LAB2_DIR="$(cd "$CODE_DIR/.." && pwd)"
ENV_YAML="$CODE_DIR/environment.yaml"
ENV_NAME="env_214"
DATA_DIR="$LAB2_DIR/data"
RESULTS_DIR="$CODE_DIR/results"

echo "[INFO] CODE_DIR   = $CODE_DIR"
echo "[INFO] LAB2_DIR   = $LAB2_DIR"
echo "[INFO] ENV_YAML   = $ENV_YAML"
echo "[INFO] DATA_DIR   = $DATA_DIR"
echo "[INFO] RESULTS_DIR= $RESULTS_DIR"

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






# ========= convert npz float64 -> float32 =========
echo "[INFO] Running float32 conversion on $DATA_DIR ..."
python - <<'PY'
import numpy as np
from pathlib import Path

src_dir = Path("../data")
dst_dir = Path("../data")
dst_dir.mkdir(parents=True, exist_ok=True)

npz_files = sorted(src_dir.glob("*.npz"))

if len(npz_files) == 0:
    print("No npz files found in", src_dir)
else:
    all_mean_abs = []
    all_max_abs = []
    all_mean_rel = []

    for f in npz_files:
        data = np.load(f)

        save_dict = {}
        file_mean_abs_list = []
        file_max_abs_list = []
        file_mean_rel_list = []

        for key in data.files:
            arr64 = data[key]

            if np.issubdtype(arr64.dtype, np.floating):
                arr32 = arr64.astype(np.float32)

                abs_diff = np.abs(arr64 - arr32.astype(arr64.dtype))
                mean_abs = float(abs_diff.mean())
                max_abs = float(abs_diff.max())

                rel_diff = abs_diff / (np.abs(arr64) + 1e-12)
                mean_rel = float(rel_diff.mean())

                file_mean_abs_list.append(mean_abs)
                file_max_abs_list.append(max_abs)
                file_mean_rel_list.append(mean_rel)

                all_mean_abs.append(mean_abs)
                all_max_abs.append(max_abs)
                all_mean_rel.append(mean_rel)

                save_dict[key] = arr32
            else:
                save_dict[key] = arr64

        out_path = dst_dir / f.name
        np.savez_compressed(out_path, **save_dict)

    if all_mean_abs:
        print("\n=== Overall summary ===")
        print(f"number of files: {len(npz_files)}")
        print(f"overall mean(abs error): {np.mean(all_mean_abs):.10g}")
        print(f"overall max(abs error):  {np.max(all_max_abs):.10g}")
        print(f"overall mean(rel error): {np.mean(all_mean_rel):.10g}")
    else:
        print("\nNo floating-point arrays were found in these npz files.")
PY






# ========= transfer learning =========
echo "[INFO] Submitting transfer learning jobs..."

PRETRAIN_JOBID=$(
    sbatch job.sh configs/pretrain.yaml | awk '{print $4}'
)
echo "[INFO] pretrain job id: $PRETRAIN_JOBID"

FINETUNE_CV_JOBID=$(
    sbatch --dependency=afterok:"$PRETRAIN_JOBID" job.sh configs/finetune_cv.yaml | awk '{print $4}'
)
echo "[INFO] finetune_cv job id: $FINETUNE_CV_JOBID"

FINETUNE_FINAL_JOBID=$(
    sbatch --dependency=afterok:"$FINETUNE_CV_JOBID" job.sh configs/finetune_final.yaml | awk '{print $4}'
)
echo "[INFO] finetune_final job id: $FINETUNE_FINAL_JOBID"







# ========= Model A input: extract latent vectors =========
EXTRACT_CMD=$(cat <<'EOF'
set -euo pipefail
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env_214
cd "'"$CODE_DIR"'"
python extract_part3_latent_vectors.py \
  configs/finetune_final.yaml \
  results/transfer_learning/modified/finetune/final/final-epoch=004-v2.ckpt \
  results/part3_latent_vectors.npz
EOF
)

EXTRACT_JOBID=$(
    sbatch --dependency=afterok:"$FINETUNE_FINAL_JOBID" \
    --job-name=extract_latent \
    --output=results/slurm-extract-latent-%j.out \
    --wrap "$EXTRACT_CMD" | awk '{print $4}'
)
echo "[INFO] extract_part3_latent_vectors job id: $EXTRACT_JOBID"








# ========= ModelA : random forest =========
RF_CMD=$(cat <<'EOF'
set -euo pipefail
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env_214
cd "'"$CODE_DIR"'"
python random_forest/part3_random_forest.py \
  --ae-features results/part3_latent_vectors.npz \
  --labeled-paths \
    ../data/O012791.npz \
    ../data/O013257.npz \
    ../data/O013490.npz \
  --outdir results/part3_random_forest \
  --random-state 42
EOF
)

RF_JOBID=$(
    sbatch --dependency=afterok:"$EXTRACT_JOBID" \
    --job-name=part3_rf \
    --output=results/slurm-part3-rf-%j.out \
    --wrap "$RF_CMD" | awk '{print $4}'
)
echo "[INFO] random forest job id: $RF_JOBID"

echo "[INFO] Pipeline submitted successfully."
echo "[INFO] Job chain:"
echo "        pretrain        : $PRETRAIN_JOBID"
echo "        finetune_cv     : $FINETUNE_CV_JOBID"
echo "        finetune_final  : $FINETUNE_FINAL_JOBID"
echo "        extract_latent  : $EXTRACT_JOBID"
echo "        random_forest   : $RF_JOBID"