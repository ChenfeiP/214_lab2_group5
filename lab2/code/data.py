import glob
import numpy as np
import os 

def load_norm(path):
    norm = np.load(path)
    return {
        "means": norm["means"],
        "stds": norm["stds"],
    }

def save_norm(norm, path):
    np.savez(path, means=norm["means"], stds=norm["stds"])

def make_data(patch_size=9, path="../image_data_float32/*.npz", norm=None, return_norm=False,):
    """
    Load the image data and create patches from it.
    Args:
        patch_size: The size of the patches to create.
    Returns:
        images_long: A list of numpy arrays of the original images.
        patches: A list of lists of patches for each image.
    """
    # 1. load data
    path = glob.glob(path)
    images_long = []
    for fp in path:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]  # remove labels
        images_long.append(data)   # it is a list of each image, each row is a pixel

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
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny    # Use global minimums to get relative coordinates.
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width))    # (feature, y range, x range)-value for each feature, we have a big zero matrix to store the data
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    # 4. convert to a 4D array.
    
    images = np.array(images)    # it is (number of images, feature, height, width)-value
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
    for i in range(len(images_long)):   # loop for images
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        # Pad the image by reflecting across the border.
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        # Use global min values to compute relative indices.
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[      # (feature, y, x)-value. here is (8,9,9), which means 8 features, for each features we have 9x9 values.
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float32))
        patches.append(patches_img)   # patches[i][j] = i-th images, j-th pixel pair(x[j],y[j]), we have 8 features, each features we have 91 (8,9,9)-value

    if return_norm:
        return images_long, patches, norm
    return images_long, patches
