# EXAMPLE USAGE:
# python run_autoencoder.py configs/pretrain.yaml
# python run_autoencoder.py configs/finetune.yaml

import numpy as np
import sys
import os
import yaml  # pip install pyyaml
import gc
import torch
import lightning as L
import random

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# pip install torchinfo
# from torchinfo import summary

from autoencoder import Autoencoder
from patchdataset import PatchDataset
from data import make_data, save_norm, load_norm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_files(path, train_fraction=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(path))
    rng.shuffle(idx)

    n_train = max(1, int(train_fraction * len(path)))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_files = [path[i] for i in train_idx]
    val_files = [path[i] for i in val_idx]
    return train_files, val_files


def flatten_patches(patches):
    all_patches = [patch for image_patches in patches for patch in image_patches]
    return all_patches

print("loading config file")
config_path = sys.argv[1]
assert os.path.exists(config_path), f"Config file {config_path} not found"
config = yaml.safe_load(open(config_path, "r"))

seed = config.get("seed", 42)
set_seed(seed)

# clean up memory
gc.collect()
torch.cuda.empty_cache()

stage = config["stage"]
assert stage in ["pretrain", "finetune"], "stage must be 'pretrain' or 'finetune'"

print(f"running stage = {stage}")

if stage == "pretrain":
    # use unlabeled images
    filepaths = config["data"]["pretrain_filepaths"]

    # image-level split
    train_files, val_files = split_files(
        filepaths,
        train_fraction=config["data"].get("train_fraction", 0.8),
        seed=seed,
    )

    print("making train patch data for pretraining")
    _, train_patches_nested, norm = make_data(
        patch_size=config["data"]["patch_size"],
        filepaths=train_files,
        norm=None,
        return_norm=True,
    )

    print("making val patch data for pretraining")
    _, val_patches_nested = make_data(
        patch_size=config["data"]["patch_size"],
        filepaths=val_files,
        norm=norm,
        return_norm=False,
    )

    # save normalization stats
    norm_path = config["data"]["norm_save_path"]
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    save_norm(norm, norm_path)
    print(f"saved norm stats to {norm_path}")

    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

elif stage == "finetune":
    # use labeled training images only
    filepaths = config["data"]["finetune_filepaths"]

    # image-level split
    train_files, val_files = split_files(
        filepaths,
        train_fraction=config["data"].get("train_fraction", 0.8),
        seed=seed,
    )

    # load normalization stats from pretraining
    norm_path = config["data"]["norm_load_path"]
    norm = load_norm(norm_path)
    print(f"loaded norm stats from {norm_path}")

    print("making train patch data for finetuning")
    _, train_patches_nested = make_data(
        patch_size=config["data"]["patch_size"],
        filepaths=train_files,
        norm=norm,
        return_norm=False,
    )

    print("making val patch data for finetuning")
    _, val_patches_nested = make_data(
        patch_size=config["data"]["patch_size"],
        filepaths=val_files,
        norm=norm,
        return_norm=False,
    )

    print("loading pretrained checkpoint")
    pretrained_ckpt = config["data"]["pretrained_checkpoint_path"]
    model = Autoencoder.load_from_checkpoint(
        pretrained_ckpt,
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

train_patches = flatten_patches(train_patches_nested)
val_patches = flatten_patches(val_patches_nested)

print(f"num train patches: {len(train_patches)}")
print(f"num val patches: {len(val_patches)}")

train_dataset = PatchDataset(train_patches)
val_dataset = PatchDataset(val_patches)

# create train and val dataloaders
dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

print("initializing model")
# Initialize an autoencoder object
model = Autoencoder(
    optimizer_config=config["optimizer"],
    patch_size=config["data"]["patch_size"],
    **config["autoencoder"],
)
print(model)
# print(summary(model, (8, 9, 9)))

print("preparing for training")
# configure the settings for making checkpoints
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# if running in slurm, add slurm job id info to the config file
if "SLURM_JOB_ID" in os.environ:
    config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

# initialize the wandb logger, giving it our config file
# to save, and also configuring the logger itself.
wandb_logger = WandbLogger(config=config, **config["wandb"])

# initialize the trainer
trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)

print("training")
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

# clean up memory
gc.collect()
torch.cuda.empty_cache()
