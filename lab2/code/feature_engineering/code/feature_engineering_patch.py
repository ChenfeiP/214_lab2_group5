#!/usr/bin/env python3
import numpy as np
import pandas as pd


def build_spatial_grid(image_df, value_col, x_col="x", y_col="y"):
    """
    Convert one image from long-form rows into a 2D grid.
    Parameters:
    image_df : pd.DataFrame
        DataFrame with columns x, y, and value_col.
    value_col : str
        Column name of the value to convert to a grid.
    x_col : str
        Column name of the x coordinate.
    y_col : str
        Column name of the y coordinate.

    Returns:
        grid : np.ndarray
            2D grid with NaN for missing coordinates.
        x_values, y_values : np.ndarray
            Sorted unique coordinate values.
        x_to_idx, y_to_idx : dict
            Maps original coordinates to grid indices.
    """
    temp = image_df[[x_col, y_col, value_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")

    x_values = np.sort(temp[x_col].dropna().unique())
    y_values = np.sort(temp[y_col].dropna().unique())

    x_to_idx = {x: i for i, x in enumerate(x_values)}
    y_to_idx = {y: i for i, y in enumerate(y_values)}

    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

    valid_rows = temp[temp[x_col].notna() & temp[y_col].notna()]
    for x_val, y_val, feature_val in valid_rows[[x_col, y_col, value_col]].to_numpy():
        row_idx = y_to_idx[y_val]
        col_idx = x_to_idx[x_val]
        grid[row_idx, col_idx] = feature_val

    return grid, x_values, y_values, x_to_idx, y_to_idx


def compute_local_mean_std(grid, window_size=3):
    """
    Compute NaN-aware local mean and std for each cell in a 2D grid.

    Border pixels use only the available neighbors inside the image bounds.

    NaNs are ignored within each local window. If all values are NaN, the output
    for that pixel is NaN.

    Returns:
        local_mean : np.ndarray
            2D grid with local mean for each cell.
        local_std : np.ndarray
            2D grid with local std for each cell.

    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    radius = window_size // 2
    n_rows, n_cols = grid.shape

    local_mean = np.full_like(grid, np.nan, dtype=float)
    local_std = np.full_like(grid, np.nan, dtype=float)

    for r in range(n_rows):
        r0 = max(0, r - radius)
        r1 = min(n_rows, r + radius + 1)

        for c in range(n_cols):
            c0 = max(0, c - radius)
            c1 = min(n_cols, c + radius + 1)

            window = grid[r0:r1, c0:c1]
            valid = window[~np.isnan(window)]

            if valid.size > 0:
                local_mean[r, c] = np.mean(valid)
                local_std[r, c] = np.std(valid, ddof=0)

    return local_mean, local_std


def engineer_patch_features_for_image(
    image_df,
    feature_cols=("NDAI", "CORR", "SD"),
    x_col="x",
    y_col="y",
):
    """
    Create 3x3 local mean/std engineered features for one image_id.

    Adds, for each feature F:
    - F_mean_3x3
    - F_std_3x3
    """
    image_df_new = image_df.copy()

    for feature in feature_cols:
        grid, _, _, x_to_idx, y_to_idx = build_spatial_grid(
            image_df_new,
            value_col=feature,
            x_col=x_col,
            y_col=y_col,
        )

        local_mean_grid, local_std_grid = compute_local_mean_std(grid, window_size=3)

        mean_col = f"{feature}_mean_3x3"
        std_col = f"{feature}_std_3x3"

        engineered_means = []
        engineered_stds = []

        for x_val, y_val in image_df_new[[x_col, y_col]].to_numpy():
            if pd.isna(x_val) or pd.isna(y_val):
                engineered_means.append(np.nan)
                engineered_stds.append(np.nan)
                continue

            row_idx = y_to_idx.get(y_val)
            col_idx = x_to_idx.get(x_val)

            if row_idx is None or col_idx is None:
                engineered_means.append(np.nan)
                engineered_stds.append(np.nan)
            else:
                engineered_means.append(local_mean_grid[row_idx, col_idx])
                engineered_stds.append(local_std_grid[row_idx, col_idx])

        image_df_new[mean_col] = engineered_means
        image_df_new[std_col] = engineered_stds

    return image_df_new


def engineer_patch_features(
    df,
    feature_cols=("NDAI", "CORR", "SD"),
    image_id_col="image_id",
    x_col="x",
    y_col="y",
):
    """
    Process each image separately and add 3x3 patch engineered features.

    Returns a new DataFrame with all original columns plus:
    - NDAI_mean_3x3, NDAI_std_3x3
    - CORR_mean_3x3, CORR_std_3x3
    - SD_mean_3x3, SD_std_3x3
    """
    required_cols = [image_id_col, x_col, y_col] + list(feature_cols)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    pieces = []
    for _, image_df in df.groupby(image_id_col, sort=False):
        image_df_new = engineer_patch_features_for_image(
            image_df=image_df,
            feature_cols=feature_cols,
            x_col=x_col,
            y_col=y_col,
        )
        pieces.append(image_df_new)

    df_new = pd.concat(pieces, axis=0)
    df_new = df_new.loc[df.index]
    return df_new


def main_example():
    """
    Example usage.
    Replace this with your own loading code if needed.
    """
    
    # df_new = engineer_patch_features(df, feature_cols=["NDAI", "CORR", "SD"])
    # print(df_new.columns)
    # print(df_new.head())
    pass


if __name__ == "__main__":
    print("Example usage:")
    print('df_new = engineer_patch_features(df, feature_cols=["NDAI", "CORR", "SD"])')
    print("print(df_new.columns)")
    print("print(df_new.head())")
