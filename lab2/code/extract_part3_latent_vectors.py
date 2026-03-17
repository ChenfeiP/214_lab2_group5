# EXAMPLE USAGE:
# python extract_part3_latent_vectors.py \
#   configs/finetune_final.yaml \
#   checkpoints/finetune/final/final-epoch=004.ckpt \
#   results/part3_latent_vectors.npz

import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autoencoder import Autoencoder
from data import make_data, load_norm


class PatchFeatureDataset(Dataset):
    def __init__(self, patches, labels, groups):
        self.patches = patches
        self.labels = labels
        self.groups = groups

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.tensor(self.patches[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        g = torch.tensor(self.groups[idx], dtype=torch.long)
        return x, y, g


def load_labels_from_npz(npz_path):
    npz_data = np.load(npz_path)
    key = list(npz_data.files)[0]
    data = npz_data[key]

    if data.shape[1] != 11:
        raise ValueError(
            f"{npz_path} does not appear to contain labels. "
            f"Expected 11 columns, got {data.shape[1]}."
        )

    labels = data[:, -1].astype(int)

    # 只保留有 expert label 的点
    keep = labels != 0
    labels = labels[keep]

    # +1 -> 1 (cloud), -1 -> 0 (non-cloud)
    labels01 = (labels == 1).astype(np.int64)
    return labels01


def main():
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: python extract_part3_latent_vectors.py "
            "<config_path> <checkpoint_path> <output_npz>"
        )

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_path = sys.argv[3]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    labeled_paths = config["data"]["finetune_path"]
    patch_size = config["data"]["patch_size"]
    norm_path = config["data"]["norm_load_path"]

    norm = load_norm(norm_path)
    print(f"Loaded norm stats from: {norm_path}")

    # make_data 会删掉最后一列 label，但 patch 顺序与原始行顺序一致
    _, patches_nested = make_data(
        patch_size=patch_size,
        path=labeled_paths,
        norm=norm,
        return_norm=False,
    )

    all_patches = []
    all_labels = []
    all_groups = []
    image_names = []

    for group_id, (img_path, img_patches) in enumerate(zip(labeled_paths, patches_nested)):
        img_name = os.path.basename(img_path)
        image_names.append(img_name)

        img_labels = load_labels_from_npz(img_path)

        if len(img_patches) != len(img_labels):
            raise ValueError(
                f"Mismatch for {img_name}: "
                f"{len(img_patches)} patches vs {len(img_labels)} labels."
            )

        all_patches.extend(img_patches)
        all_labels.extend(img_labels.tolist())
        all_groups.extend([group_id] * len(img_patches))

        print(f"{img_name}: {len(img_patches)} labeled patches")

    dataset = PatchFeatureDataset(all_patches, all_labels, all_groups)
    loader = DataLoader(
        dataset,
        batch_size=config["dataloader_val"].get("batch_size", 1024),
        shuffle=False,
        num_workers=config["dataloader_val"].get("num_workers", 0),
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    model = Autoencoder.load_from_checkpoint(
        checkpoint_path,
        optimizer_config=config["optimizer"],
        patch_size=patch_size,
        **config["autoencoder"],
    )
    model.eval()
    model.to(device)

    X_list = []
    y_list = []
    g_list = []

    with torch.no_grad():
        for batch_x, batch_y, batch_g in loader:
            batch_x = batch_x.to(device)

            z = model.embed(batch_x)
            z = z.view(z.size(0), -1)

            X_list.append(z.cpu().numpy())
            y_list.append(batch_y.numpy())
            g_list.append(batch_g.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    image_names = np.array(image_names, dtype=object)

    out_dir = os.path.dirname(output_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        output_path,
        X=X,
        y=y,
        groups=groups,
        image_names=image_names,
    )

    print(f"Saved latent vectors to: {output_path}")
    print(f"X shape      : {X.shape}")
    print(f"y shape      : {y.shape}")
    print(f"groups shape : {groups.shape}")
    print(f"image_names  : {image_names}")


if __name__ == "__main__":
    main()