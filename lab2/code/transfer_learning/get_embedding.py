# EXAMPLE (cwd = lab2/code):
#   python transfer_learning/get_embedding.py \\
#     transfer_learning/configs/finetune_final.yaml \\
#     results/transfer_learning/checkpoints_modified/finetune/final/final-004.ckpt
#
# Baseline checkpoints/embeddings: use configs/finetune_final_baseline.yaml and
#   results/transfer_learning/checkpoints_baseline/finetune/...

import sys
import os

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
import torch
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from autoencoder import Autoencoder
from data import make_data, load_norm

config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

config = yaml.safe_load(open(config_path, "r"))

print("Loading the saved model")

# initialize the autoencoder class
model = Autoencoder(
    patch_size=config["data"]["patch_size"],
    **config["autoencoder"],
)
# tell PyTorch to load the model onto the CPU if no GPU is available
map_location = None if torch.cuda.is_available() else "cpu"
# load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=map_location)
# load the checkpoint's state_dict into the model
model.load_state_dict(checkpoint["state_dict"])
# put the model in evaluation mode
model.eval()

embedding_size = config["autoencoder"].get("embedding_size", 8)
norm = None
norm_path = config["data"].get("norm_load_path")
path = config["data"].get("embedding_path") or config["data"].get("finetune_path") or config["data"].get("pretrain_path")
if path is None:
    raise ValueError("config must specify one of: data.embedding_path, data.finetune_path, data.pretrain_path")
if norm_path and os.path.exists(norm_path):
    norm = load_norm(norm_path)
    print(f"Using normalization from {norm_path}")

print("Making the patch data")
images_long, patches = make_data(
    patch_size=config["data"]["patch_size"],
    path=path,
    norm=norm,
)

print("Obtaining embeddings")
# get the embedding for each patch
embeddings = []  # what we will save

for i in tqdm(range(len(images_long))):
    ys = images_long[i][:, 0]
    xs = images_long[i][:, 1]
    # determine the height and width of the image
    miny, minx = min(ys), min(xs)
    height = int(max(ys) - miny + 1)
    width = int(max(xs) - minx + 1)

    # to make this faster, we use torch.no_grad() to disable gradient tracking
    with torch.no_grad():
        # get the embedding of array of patches
        emb = model.embed(torch.tensor(np.array(patches[i])))
        # NOTE: if your model is quite big, you may not be able to fit
        # all of the data into the GPU memory at once for inference.
        # In that case, you can loop over smaller bathches of data.

        # in the following line we:
        # - detach the tensor from the computation graph
        # - move it to the cpu
        # - turn it into a numpy array
        emb = emb.detach().cpu().numpy()

    embeddings.append(emb)

print("Saving the embeddings")
# save the embeddings as csv
output_dir = config["data"].get(
    "embedding_output_dir", "results/transfer_learning/results_modified"
)
os.makedirs(output_dir, exist_ok=True)
for i in tqdm(range(len(images_long))):
    embedding_df = pd.DataFrame(
        embeddings[i],
        columns=[f"ae{j}" for j in range(embedding_size)],
    )
    # add y and x to the dataframe
    embedding_df["y"] = images_long[i][:, 0]
    embedding_df["x"] = images_long[i][:, 1]
   
    cols = embedding_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    embedding_df = embedding_df[cols]
     # save to csv
    embedding_df.to_csv(
        os.path.join(output_dir, f"image{i+1}_ae.csv"),
        index=False,
    )
print(f"Saved embedding CSVs to {output_dir}/")
