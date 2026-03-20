# From lab2/code (cwd must be lab2/code so ../data and results/ resolve correctly):
#
# Modified (default) checkpoints under results/transfer_learning/checkpoints_modified/:
#   python run_autoencoder.py transfer_learning/configs/pretrain.yaml
#   python run_autoencoder.py transfer_learning/configs/finetune_cv.yaml
#   python run_autoencoder.py transfer_learning/configs/finetune_final.yaml
#
# Baseline:
#   python run_autoencoder.py transfer_learning/configs/pretrain_baseline.yaml
#   python run_autoencoder.py transfer_learning/configs/finetune_cv_baseline.yaml
#   python run_autoencoder.py transfer_learning/configs/finetune_final_baseline.yaml

import numpy as np
import sys
import os
import yaml
import gc
import copy
import torch
torch.set_float32_matmul_precision("medium")
import lightning as L
import random
import glob

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

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


def make_leave_one_out_folds(path):
    folds = []
    n = len(path)
    for val_idx in range(n):
        val_files = [path[val_idx]]
        train_files = [path[i] for i in range(n) if i != val_idx]
        folds.append((train_files, val_files))
    return folds


def flatten_patches(patches):
    return [patch for image_patches in patches for patch in image_patches]


def set_all_trainable(model):
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Trainable params: {trainable_params}/{total_params} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    print("Trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")


def get_pretrained_ckpt(config):
    ckpt_path = config["data"].get("pretrained_checkpoint_path", None)
    if ckpt_path is not None:
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        return ckpt_path

    ckpt_dir = config["data"]["pretrained_checkpoint_dir"]
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    assert ckpt_files, f"No checkpoint found in {ckpt_dir}"
    return ckpt_files[0]


print("======= Step1: loading config file =======")
config_path = sys.argv[1]
assert os.path.exists(config_path), f"Config file {config_path} not found"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

seed = config.get("seed", 42)
set_seed(seed)

gc.collect()
torch.cuda.empty_cache()

stage = config["stage"]
assert stage in ["pretrain", "finetune_cv", "finetune_final"], \
    "stage must be 'pretrain', 'finetune_cv', or 'finetune_final'"

print(f"======= Step2: running stage = {stage} =======")

if stage == "pretrain":
    path = config["data"]["pretrain_path"]

    train_files, val_files = split_files(
        path,
        train_fraction=config["data"].get("train_fraction", 0.8),
        seed=seed,
    )

    print("making train patch data for pretraining")
    _, train_patches_nested, norm = make_data(
        patch_size=config["data"]["patch_size"],
        path=train_files,
        norm=None,
        return_norm=True,
    )

    print("making val patch data for pretraining")
    _, val_patches_nested = make_data(
        patch_size=config["data"]["patch_size"],
        path=val_files,
        norm=norm,
        return_norm=False,
    )

    norm_path = config["data"]["norm_save_path"]
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    save_norm(norm, norm_path)
    print(f"saved norm stats to {norm_path}")

    model = Autoencoder(
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

    dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
    dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

    del train_patches_nested, val_patches_nested, train_patches, val_patches
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

    use_wandb = config.get("use_wandb", True)
    wandb_logger = WandbLogger(config=config, **config["wandb"]) if use_wandb else None

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **config["trainer"],
    )

    print("======= Step3: training =======")
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)


elif stage == "finetune_cv":
    path = config["data"]["finetune_path"]
    assert len(path) == 3, "finetune_cv currently expects exactly 3 images"

    norm_path = config["data"]["norm_load_path"]
    norm = load_norm(norm_path)
    print(f"loaded norm stats from {norm_path}")

    pretrained_ckpt = get_pretrained_ckpt(config)
    print(f"loading pretrained checkpoint: {pretrained_ckpt}")

    folds = make_leave_one_out_folds(path)
    fold_best_val_losses = []
    base_ckpt_dir = config["checkpoint"]["dirpath"]

    for fold_idx, (train_files, val_files) in enumerate(folds):
        print("\n" + "=" * 60)
        print(f"Starting fold {fold_idx + 1}/{len(folds)}")
        print(f"train_files = {train_files}")
        print(f"val_files   = {val_files}")
        print("=" * 60)

        set_seed(seed + fold_idx)

        print("making train patch data for finetuning")
        _, train_patches_nested = make_data(
            patch_size=config["data"]["patch_size"],
            path=train_files,
            norm=norm,
            return_norm=False,
        )

        print("making val patch data for finetuning")
        _, val_patches_nested = make_data(
            patch_size=config["data"]["patch_size"],
            path=val_files,
            norm=norm,
            return_norm=False,
        )

        train_patches = flatten_patches(train_patches_nested)
        val_patches = flatten_patches(val_patches_nested)

        print(f"num train patches: {len(train_patches)}")
        print(f"num val patches: {len(val_patches)}")

        train_dataset = PatchDataset(train_patches)
        val_dataset = PatchDataset(val_patches)

        dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
        dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

        del train_patches_nested, val_patches_nested, train_patches, val_patches
        gc.collect()
        torch.cuda.empty_cache()

        model = Autoencoder.load_from_checkpoint(
            pretrained_ckpt,
            optimizer_config=config["optimizer"],
            patch_size=config["data"]["patch_size"],
            **config["autoencoder"],
        )

        set_all_trainable(model)

        fold_checkpoint_config = copy.deepcopy(config["checkpoint"])
        fold_checkpoint_config["dirpath"] = os.path.join(base_ckpt_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_checkpoint_config["dirpath"], exist_ok=True)
        checkpoint_callback = ModelCheckpoint(**fold_checkpoint_config)

        use_wandb = config.get("use_wandb", True)
        if use_wandb:
            fold_wandb_config = copy.deepcopy(config)
            fold_wandb_config["fold"] = fold_idx + 1
            fold_wandb_kwargs = copy.deepcopy(config["wandb"])
            fold_wandb_kwargs["name"] = f'{config["wandb"]["name"]}-fold{fold_idx+1}'
            wandb_logger = WandbLogger(config=fold_wandb_config, **fold_wandb_kwargs)
        else:
            wandb_logger = None

        trainer = L.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            **config["trainer"],
        )

        print(f"======= training fold {fold_idx + 1} =======")
        trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

        best_val = checkpoint_callback.best_model_score
        if best_val is not None:
            best_val = best_val.item()
            fold_best_val_losses.append(best_val)
            print(f"Fold {fold_idx + 1} best val_loss = {best_val:.6f}")
            print(f"Fold {fold_idx + 1} best checkpoint = {checkpoint_callback.best_model_path}")
        else:
            print(f"Fold {fold_idx + 1}: best_model_score is None")

        del model, train_dataset, val_dataset, dataloader_train, dataloader_val, trainer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Leave-one-out cross validation finished")
    for i, loss in enumerate(fold_best_val_losses):
        print(f"Fold {i+1} best val_loss: {loss:.6f}")

    if len(fold_best_val_losses) == len(folds):
        mean_val_loss = np.mean(fold_best_val_losses)
        std_val_loss = np.std(fold_best_val_losses)
        print(f"Mean best val_loss: {mean_val_loss:.6f}")
        print(f"Std  best val_loss: {std_val_loss:.6f}")
    else:
        print("Warning: not all folds produced a valid best val_loss")
    print("=" * 60)


elif stage == "finetune_final":
    path = config["data"]["finetune_path"]
    assert len(path) >= 1, "Need at least 1 image for finetuning"

    norm_path = config["data"]["norm_load_path"]
    norm = load_norm(norm_path)
    print(f"loaded norm stats from {norm_path}")

    print("making full training patch data for final finetuning")
    _, train_patches_nested = make_data(
        patch_size=config["data"]["patch_size"],
        path=path,
        norm=norm,
        return_norm=False,
    )

    train_patches = flatten_patches(train_patches_nested)
    print(f"num train patches: {len(train_patches)}")

    train_dataset = PatchDataset(train_patches)
    dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])

    del train_patches_nested, train_patches
    gc.collect()
    torch.cuda.empty_cache()

    pretrained_ckpt = get_pretrained_ckpt(config)
    print(f"loading pretrained checkpoint: {pretrained_ckpt}")

    model = Autoencoder.load_from_checkpoint(
        pretrained_ckpt,
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    set_all_trainable(model)

    final_checkpoint_config = copy.deepcopy(config["checkpoint"])
    final_checkpoint_config["dirpath"] = os.path.join(config["checkpoint"]["dirpath"], "final")
    os.makedirs(final_checkpoint_config["dirpath"], exist_ok=True)

    final_checkpoint_config.pop("monitor", None)
    final_checkpoint_config.pop("mode", None)
    final_checkpoint_config["save_top_k"] = -1
    final_checkpoint_config["filename"] = "final-{epoch:03d}"

    checkpoint_callback = ModelCheckpoint(**final_checkpoint_config)

    use_wandb = config.get("use_wandb", True)
    if use_wandb:
        final_wandb_config = copy.deepcopy(config)
        final_wandb_config["stage"] = "finetune_final"
        final_wandb_kwargs = copy.deepcopy(config["wandb"])
        final_wandb_kwargs["name"] = f'{config["wandb"]["name"]}-final'
        wandb_logger = WandbLogger(config=final_wandb_config, **final_wandb_kwargs)
    else:
        wandb_logger = None

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **config["trainer"],
    )

    print("======= final finetuning on all images =======")
    trainer.fit(model, train_dataloaders=dataloader_train)

    print("Final finetuning finished")
    print(f"Checkpoints saved to: {final_checkpoint_config['dirpath']}")

gc.collect()
torch.cuda.empty_cache()