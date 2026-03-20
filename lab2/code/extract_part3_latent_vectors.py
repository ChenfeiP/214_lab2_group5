

import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autoencoder import Autoencoder
from data import make_data_part3, load_norm


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


def resolve_path(p):
    if isinstance(p, str) and not os.path.isabs(p):
        return os.path.normpath(os.path.join(os.getcwd(), p))
    return p


def main():
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: python extract_part3_latent_vectors.py "
            "<config_path> <checkpoint_path> <output_npz>"
        )

    config_path = sys.argv[1]
    checkpoint_path = resolve_path(sys.argv[2])
    output_path = resolve_path(sys.argv[3])

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    labeled_paths = [resolve_path(p) for p in config["data"]["finetune_path"]]
    patch_size = config["data"]["patch_size"]
    norm_path = resolve_path(config["data"]["norm_load_path"])

    print(f"Current working directory: {os.getcwd()}")
    print(f"Using config: {config_path}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using norm stats: {norm_path}")
    print(f"Using labeled paths:")
    for p in labeled_paths:
        print(f"  {p}")

    norm = load_norm(norm_path)
    print(f"Loaded norm stats from: {norm_path}")


    _, patches_nested, labels_nested, groups_nested, image_names = make_data_part3(
        patch_size=patch_size,
        path=labeled_paths,
        norm=norm,
        return_norm=False,
    )

    all_patches = np.concatenate(patches_nested, axis=0)
    all_labels = np.concatenate(labels_nested, axis=0)
    all_groups = np.concatenate(groups_nested, axis=0)

    print(f"Total labeled patches: {len(all_patches)}")
    print(f"Label shape: {all_labels.shape}")
    print(f"Group shape: {all_groups.shape}")
    print(f"Image names: {image_names}")

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