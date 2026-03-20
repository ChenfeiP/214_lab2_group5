import glob
import numpy as np
import os 
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view 

def load_norm(path):
    norm = np.load(path)
    return {
        "means": norm["means"],
        "stds": norm["stds"],
    }

def save_norm(norm, path):
    np.savez(path, means=norm["means"], stds=norm["stds"])

def make_data(patch_size=9, path="../image_data_float32/*.npz", norm=None, 
              return_norm=False, remove_labels=True):
    """
    Load the image data and create patches from it.
    Args:
        patch_size: The size of the patches to create.
        path: The filepath from which to get the float32 images.
        norm: The norm to use.
        return_norm: Whether to return the norm.
        remove_labels: Whether to remove labels (Note: If we don't
        remove labels, we return patches as an empty list).

    Returns:
        images_long: A list of numpy arrays of the original images.
        patches: A list of lists of patches for each image.
    """    
    # 1. load data
    if isinstance(path, str):
        path = sorted(glob.glob(path))
    elif isinstance(path, list):
        path = sorted(path)
    else:
        raise TypeError(f"path must be a string or a list of strings, got {type(path)}")

    assert len(path) > 0, "No image files found."
    
    images_long = []
    for fp in tqdm(path):
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if remove_labels and data.shape[1] == 11:
            data = data[:, :-1]  # remove labels
        images_long.append(data)   # it is a list of each image, each row is a pixel
    
    # If keeping the labels, return the patches as an empty list.
    if not remove_labels:
        return images_long, []

    # 2. calculate y and x range
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)  #  it is a list of all y 
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)  #  it is a list of all x 
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)     
    width = int(global_maxx - global_minx + 1)

    # 3. convert to (feature, y, x)-value form, ensure each has the same shape.
    nchannels = images_long[0].shape[1] - 2    # feature = 8
    images = []
    coords_rel = [] 
    for img in tqdm(images_long):
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny    # Use global minimums to get relative coordinates.
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width), dtype=np.float32)     # (feature, y range, x range)-value for each feature, we have a big zero matrix to store the data
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]

        image[:, y_valid, x_valid] = img_valid[:, 2:].T.astype(np.float32)

        images.append(image)
        coords_rel.append((y_valid, x_valid))
    print('done reshaping images')

    # 4. convert to a 4D array.
    
    images = np.array(images, dtype=np.float32)   # it is (number of images, feature, height, width)-value
    pad_len = patch_size // 2

    if norm is None:
        means = np.mean(images, axis=(0, 2, 3))[:, None, None]
        stds = np.std(images, axis=(0, 2, 3))[:, None, None]
        stds[stds == 0] = 1.0
        norm = {
            "means": means.astype(np.float32),
            "stds": stds.astype(np.float32),
        }
    else:
        means = norm["means"]
        stds = norm["stds"]
    
    images = (images - means) / stds   # it is (number of images, feature, height, width)-value

    patches = []
    for i in tqdm(range(len(images_long))):   # loop for images
        if i % 10 == 0:
            print(f'working on image {i}')
       
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )

        y_rel, x_rel = coords_rel[i]
      
        windows = sliding_window_view(
            img_mirror,
            window_shape=(patch_size, patch_size),
            axis=(1, 2),
        )
        patches_img = windows[:, y_rel, x_rel, :, :].transpose(1, 0, 2, 3).astype(np.float32)
        patches.append(patches_img)   # patches[i][j] = i-th images, j-th pixel pair(x[j],y[j]), we have 8 features, each features we have 91 (8,9,9)-value

    if return_norm:
        return images_long, patches, norm
    return images_long, patches

def make_data_part3(patch_size=9, path="../data/*.npz", norm=None, return_norm=False):
    if isinstance(path, str):
        path = sorted(glob.glob(path))
    elif isinstance(path, list):
        path = sorted(path)
    else:
        raise TypeError(f"path must be a string or a list of strings, got {type(path)}")

    assert len(path) > 0, "No image files found."

    images_long = []
    for fp in tqdm(path):
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]

        if data.shape[1] != 11:
            raise ValueError(
                f"{fp} does not contain labels. Expected 11 columns, got {data.shape[1]}."
            )

        images_long.append(data)

    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    nchannels = images_long[0].shape[1] - 3  
    images = []
    coords_rel = []
    labels_valid_all = []

    for img in tqdm(images_long):
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        feats = img[:, 2:-1].astype(np.float32)
        labels = img[:, -1].astype(int)

        y_rel = y - global_miny
        x_rel = x - global_minx

        image = np.zeros((nchannels, height, width), dtype=np.float32)

        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        feats_valid = feats[valid_mask]
        labels_valid = labels[valid_mask]

        image[:, y_valid, x_valid] = feats_valid.T.astype(np.float32)

        images.append(image)
        coords_rel.append((y_valid, x_valid))
        labels_valid_all.append(labels_valid)

    print("done reshaping labeled images")

    images = np.array(images, dtype=np.float32)
    pad_len = patch_size // 2

    if norm is None:
        means = np.mean(images, axis=(0, 2, 3))[:, None, None]
        stds = np.std(images, axis=(0, 2, 3))[:, None, None]
        stds[stds == 0] = 1.0
        norm = {
            "means": means.astype(np.float32),
            "stds": stds.astype(np.float32),
        }
    else:
        means = norm["means"]
        stds = norm["stds"]

    images = (images - means) / stds

    patches = []
    labels = []
    groups = []
    image_names = []

    for i in tqdm(range(len(images_long))):
        if i % 10 == 0:
            print(f"working on labeled image {i}")

        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )

        y_rel, x_rel = coords_rel[i]
        raw_labels = labels_valid_all[i]

        windows = sliding_window_view(
            img_mirror,
            window_shape=(patch_size, patch_size),
            axis=(1, 2),
        )
        patches_img = windows[:, y_rel, x_rel, :, :].transpose(1, 0, 2, 3).astype(np.float32)

        if len(patches_img) != len(raw_labels):
            raise ValueError(
                f"Mismatch in image {path[i]}: {len(patches_img)} patches vs {len(raw_labels)} labels."
            )

        keep = raw_labels != 0
        patches_img = patches_img[keep]
        labels_img = (raw_labels[keep] == 1).astype(np.int64)

        patches.append(patches_img)
        labels.append(labels_img)
        groups.append(np.full(len(labels_img), i, dtype=np.int64))
        image_names.append(os.path.basename(path[i]))

        print(
            f"{os.path.basename(path[i])}: valid={len(raw_labels)}, labeled={len(labels_img)}"
        )

    image_names = np.array(image_names, dtype=object)

    if return_norm:
        return images_long, patches, labels, groups, image_names, norm
    return images_long, patches, labels, groups, image_names