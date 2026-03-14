#!/bin/bash

# EXAMPLE USAGE:
# sbatch job.sh configs/pretrain.yaml
# sbatch job.sh configs/finetune.yaml


#SBATCH --job-name=lab2-autoencoder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=4

set -e

if [ -z "$1" ]; then
    echo "Usage: sbatch job.sh <config_path>"
    exit 1
fi

echo "Running with config: $1"
python run_autoencoder.py "$1"
