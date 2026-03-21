# STAT 214 Lab 2 — Group 5

## Introduction

This repository contains code and workflows for **Lab 2**: analysis and modeling of **MISR multi-angle cloud imagery**. The project explores **unsupervised representation learning** with convolutional autoencoders (pretraining on a large unlabeled pool, then fine-tuning on target scenes), and **supervised evaluation** of the learned embeddings alongside raw spectral–angular features.

The main modeling threads in `lab2/code` are:

- **Feature engineering ** — rank original MISR features, generate class-comparison plots, and build patch-aware visual diagnostics under a dedicated `lab2/code/feature_engineering/` workspace.
- **Transfer learning / autoencoder pipeline** — train normalization-aware patch autoencoders, export per-pixel latent vectors for three labeled images, and run lightweight probes and visualizations.
- **Random forest (Part 3)** — extract latent features from the modified transfer-learning model and train a supervised random forest classifier on the three labeled scenes.
- **LDA** — compare `handcrafted`, `ae_only`, and `combined` feature sets with shrinkage LDA using grouped cross-validation under `lab2/code/LDA_model/`.
- **Logistic regression (Part 3)** — leave-one-image-out (LOIO) logistic models using raw bands/features only, latent features only, and their combination; compare a **baseline** vs **modified** transfer-learning setup.

Results are written into workflow-specific output directories such as `lab2/code/results/`, `lab2/code/feature_engineering/results/`, and `lab2/code/LDA_model/results/` so experiments can be traced and compared without ad hoc paths.

---

## Data

Place MISR patches as **`.npz`** files where the training configs expect them:

- **`lab2/data/`** — primary location referenced by YAML (`../data/...` from `lab2/code`).
- **`lab2/image_data_float32/`** — alternative; several analysis scripts prefer `lab2/data` if it contains `.npz` files, otherwise fall back here.

The course data distribution is described in `lab2/data/download_data.txt`.

---

## Environment

### What you need installed

| Requirement | Purpose |
|-------------|---------|
| **[Miniconda](https://docs.conda.io/en/latest/miniconda.html), Anaconda, or [Mambaforge](https://github.com/conda-forge/miniforge)** | Create the project environment from `environment.yaml`. |
| **CUDA-capable GPU + drivers** (recommended) | Training `run_autoencoder.py` and `get_embedding.py` are much faster on GPU; CPU-only may work but be slow. |
| **(Optional) Slurm** | Only if you use `run.sh` / `job.sh` on a cluster. |

### Create the conda environment

The file **`lab2/code/environment.yaml`** defines an environment named **`env_214`** (Python **3.10**) and installs dependencies through **conda** and **pip**.

From anywhere:

```bash
conda env create -f lab2/code/environment.yaml    # first time
# or update an existing env:
conda env update -f lab2/code/environment.yaml --prune
conda activate env_214
```

Activate **`env_214`** before running Python in `lab2/code`, or ensure your Slurm scripts (`job.sh`, `job_post_tl.sh`, `logistic_regression/job_lr.sh`, etc.) `conda activate` the same environment. If conda is not installed under **`~/miniconda3`**, edit the `source .../conda.sh` lines in those scripts to match your machine.

### Packages installed (via `environment.yaml`)

**Conda**

- **Python 3.10**, **pip**

**Pip** (main stack)

- **NumPy**, **pandas**, **SciPy** — arrays, tables, I/O for `.npz` / CSV.  
- **matplotlib**, **seaborn** — plots (EDA, embeddings, logistic diagnostics).  
- **scikit-learn** — probes, logistic regression, CV metrics.  
- **PyYAML** (`pyyaml`) — training/analysis configs.  
- **tqdm** — progress bars.

**Pip** (deep learning — required for autoencoder + embedding export)

- **torch**, **torchvision**, **torchaudio** — models and `get_embedding.py`.  
- **lightning** — `run_autoencoder.py` training loop.

**Pip** (optional / other workflows)

- **jupyterlab** — notebooks.  
- **lightgbm** — used elsewhere in the repo (e.g. tree models).  
- **wandb** — logging if you turn on Weights & Biases in configs.

**Minimal footprint for analysis-only:** NumPy, pandas, scikit-learn, matplotlib, and SciPy are enough for **`quick_probe`**, **`visualize_embeddings`**, **`latent_dim_table`**, **`compare_transfer_results`**, and **`logistic_regression/*`** *if embeddings already exist*. **`run_autoencoder.py`** and **`get_embedding.py`** require **PyTorch** and **Lightning** as above.

### Working directory

The canonical working directory for training configs is **`lab2/code/`** (so paths such as `../data/` and `results/` in YAML resolve correctly).

The canonical working directory for the new feature-engineering scripts is **`lab2/code/feature_engineering/code/`**. Those scripts now write only under **`lab2/code/feature_engineering/results/`**, so they do not overwrite transfer-learning or Part 3 model outputs.

---

## Reproducing the feature-engineering workflow

The feature-engineering code has been grouped into a dedicated directory:

- **Code:** `lab2/code/feature_engineering/code/`
- **Outputs:** `lab2/code/feature_engineering/results/`

Run from **`lab2/code/feature_engineering/code`**:

```bash
cd lab2/code/feature_engineering/code

# Quantitative ranking of the 8 original features
python feature_engineering.py \
  --image_dir ../../../data \
  --output_dir ../results/feature_engineering_part21

# Plots for Part 1 / Part 2 write-up
python feature_engineering_plot.py \
  --image_dir ../../../data \
  --output_dir ../results/feature_engineering_plots
```

This stage is safe to run before the transfer-learning and Part 3 pipelines because it is read-only with respect to the `.npz` inputs and writes only into `lab2/code/feature_engineering/results/`.

---

## Reproducing the transfer learning pipeline

We maintain two experimental variants — **modified** and **baseline** — with separate checkpoint trees and embedding outputs so runs never overwrite each other. Configs live in `lab2/code/transfer_learning/configs/`.

| Variant   | Config files | Checkpoint root |
|-----------|----------------|-----------------|
| **Modified** | `pretrain.yaml`, `finetune_cv.yaml`, `finetune_final.yaml` | `results/transfer_learning/checkpoints_modified/` |
| **Baseline** | `pretrain_baseline.yaml`, `finetune_cv_baseline.yaml`, `finetune_final_baseline.yaml` | `results/transfer_learning/checkpoints_baseline/` |

### Training (interactive or single node)

Always run from **`lab2/code`**:

```bash
cd lab2/code

# Modified chain
python run_autoencoder.py transfer_learning/configs/pretrain.yaml
python run_autoencoder.py transfer_learning/configs/finetune_cv.yaml
python run_autoencoder.py transfer_learning/configs/finetune_final.yaml

# Baseline chain (same stages, different configs)
python run_autoencoder.py transfer_learning/configs/pretrain_baseline.yaml
python run_autoencoder.py transfer_learning/configs/finetune_cv_baseline.yaml
python run_autoencoder.py transfer_learning/configs/finetune_final_baseline.yaml
```

`finetune_final*.yaml` sets **`embedding_output_dir`** to either `results/transfer_learning/results_modified` or `results/transfer_learning/results_baseline`. Full training checkpoints for the final stage are saved under `.../checkpoints_{variant}/finetune/final/`.

### Exporting embeddings

After the final fine-tune, choose a checkpoint (for example the latest `final-*.ckpt` under `finetune/final/`) and run:

```bash
cd lab2/code
python transfer_learning/get_embedding.py \
  transfer_learning/configs/finetune_final.yaml \
  results/transfer_learning/checkpoints_modified/finetune/final/<your_checkpoint>.ckpt
```

Use **`finetune_final_baseline.yaml`** and the **baseline** `finetune/final/` checkpoint for the baseline variant.

### Analysis and baseline-vs-modified comparison

These scripts default to sensible directories under `results/transfer_learning/`; use **`--variant modified`** or **`--variant baseline`** where applicable.

```bash
cd lab2/code
python transfer_learning/quick_probe.py --variant modified
python transfer_learning/quick_probe.py --variant baseline
python transfer_learning/latent_dim_table.py --variant modified
python transfer_learning/latent_dim_table.py --variant baseline
python transfer_learning/visualize_embeddings.py --variant modified
python transfer_learning/visualize_embeddings.py --variant baseline
python transfer_learning/compare_transfer_results.py
```

Aggregated metrics are written to **`results/transfer_learning/comparisons/transfer_learning_comparison_summary.csv`**.


---

## Reproducing the random forest experiments

Random forest uses **labeled** pixels from the three expert-annotated scenes together with latent features extracted from the **modified** transfer-learning checkpoint. Outputs go to **`results/part3_random_forest/`**.

Run from **`lab2/code`**:

```bash
cd lab2/code

python extract_part3_latent_vectors.py \
  transfer_learning/configs/finetune_final.yaml \
  results/transfer_learning/checkpoints_modified/finetune/final/final-epoch=004-v2.ckpt \
  results/part3_latent_vectors.npz

python random_forest/part3_random_forest.py \
  --ae-features results/part3_latent_vectors.npz \
  --labeled-paths \
    ../data/O012791.npz \
    ../data/O013257.npz \
    ../data/O013490.npz \
  --outdir results/part3_random_forest \
  --random-state 42
```

---

## Reproducing the LDA experiments

The LDA workflow uses the same three labeled images together with autoencoder latent vectors from the **modified** transfer-learning checkpoint. Code and outputs now live under **`lab2/code/LDA_model/`**, with results written to **`lab2/code/LDA_model/results/part3_lda/`**.

Run from **`lab2/code`**:

```bash
cd lab2/code

python extract_part3_latent_vectors.py \
  transfer_learning/configs/finetune_final.yaml \
  results/transfer_learning/checkpoints_modified/finetune/final/final-epoch=004.ckpt \
  LDA_model/results/part3_lda/part3_latent_vectors.npz

python LDA_model/LDA_model.py \
  --ae-features LDA_model/results/part3_lda/part3_latent_vectors.npz \
  --labeled-paths \
    ../data/O012791.npz \
    ../data/O013257.npz \
    ../data/O013490.npz \
  --outdir LDA_model/results/part3_lda
```

Important: the order of `--labeled-paths` should match the image order stored in the latent-vector file. Using a dedicated latent-vector path under `LDA_model/results/part3_lda/` keeps the LDA pipeline separate from the random-forest outputs, so the two workflows can be run independently without overwriting each other.

---

## Reproducing the logistic regression experiments

Logistic regression uses **labeled** pixels from the three expert-annotated scenes, merged with the embedding CSVs produced in the transfer-learning step. Outputs go to **`results/part3_logistic_regression/results_{baseline,modified}/`** (per-fold metrics, predictions, coefficients, plots, and error maps).

Run from **`lab2/code`**:

```bash
cd lab2/code
python logistic_regression/logistic_experiments.py --variant modified
python logistic_regression/logistic_experiments.py --variant baseline
python logistic_regression/compare_logistic_results.py
```

The summary table **`results/part3_logistic_regression/logistic_comparison_summary.csv`** compares baseline vs modified mean performance across feature settings (raw-only, latent-only, raw+latent).


---

## Path behavior

- **YAML + `run_autoencoder.py`:** run with **`cwd = lab2/code`** so relative paths in configs behave as intended.
- **Standalone Python tools** (`quick_probe`, `logistic_experiments`, etc.): defaults resolve **`lab2/code/results/...`** from each script’s file location, so you can often run them from other working directories and still hit the right trees; CLI flags override paths when needed.

---

## Repository layout (high level)

```
214_lab2_group5/
├── README.md                 # this file
└── lab2/
    ├── data/                 # NPZ inputs (course zip)
    ├── image_data_float32/   # optional alternate NPZ location
    └── code/
        ├── feature_engineering/
        │   ├── code/         # Part 1 / Part 2 scripts
        │   └── results/      # Part 1 / Part 2 figures + summary tables
        ├── LDA_model/
        │   ├── LDA_model.py
        │   ├── job_lda.sh
        │   └── results/      # LDA outputs
        ├── run_autoencoder.py
        ├── run.sh            # cluster driver (EDA, feature engineering, Slurm submit)
        ├── transfer_learning/  # configs, get_embedding, analysis, job_post_tl.sh
        ├── random_forest/  
        ├── logistic_regression/  # LOIO logistic, compare, job_lr.sh
        └── results/          # checkpoints, CSVs, figures (git may ignore large artifacts)
```
