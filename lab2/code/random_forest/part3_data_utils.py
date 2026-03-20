import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = {
    "y": 0,
    "x": 1,
    "NDAI": 2,
    "SD": 3,
    "CORR": 4,
    "DF": 5,
    "CF": 6,
    "BF": 7,
    "AF": 8,
    "AN": 9,
    "label": 10,
}

HANDCRAFTED_DEFAULT = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]


def _load_npz_array(npz_path: str) -> np.ndarray:
    npz_data = np.load(npz_path, allow_pickle=True)
    key = list(npz_data.files)[0]
    return npz_data[key]


def load_labeled_dataframe(
    labeled_paths: Sequence[str],
    handcrafted_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:

    if handcrafted_features is None:
        handcrafted_features = HANDCRAFTED_DEFAULT

    rows: List[pd.DataFrame] = []
    for group_id, path in enumerate(labeled_paths):
        arr = _load_npz_array(path)
        if arr.shape[1] != 11:
            raise ValueError(f"Expected 11 columns in labeled file {path}, got {arr.shape[1]}")

        df = pd.DataFrame({
            "y_coord": arr[:, FEATURE_COLUMNS["y"]].astype(int),
            "x_coord": arr[:, FEATURE_COLUMNS["x"]].astype(int),
            "image_name": os.path.basename(path),
            "group_id": group_id,
            "label_raw": arr[:, FEATURE_COLUMNS["label"]].astype(int),
        })
        for feat in handcrafted_features:
            if feat not in FEATURE_COLUMNS:
                raise KeyError(f"Unknown handcrafted feature: {feat}")
            df[feat] = arr[:, FEATURE_COLUMNS[feat]].astype(np.float32)

        df = df[df["label_raw"] != 0].copy()
        df["y_binary"] = (df["label_raw"] == 1).astype(int)
        rows.append(df.reset_index(drop=True))

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out


def load_ae_feature_dataframe(ae_feature_npz: str) -> pd.DataFrame:
    data = np.load(ae_feature_npz, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    groups = data["groups"].astype(int)
    image_names = data["image_names"]

    df = pd.DataFrame(X, columns=[f"ae_{i}" for i in range(X.shape[1])])
    df["y_binary"] = y
    df["group_id"] = groups
    df["image_name"] = [str(image_names[g]) for g in groups]
    return df


def assemble_feature_sets(
    labeled_paths: Sequence[str],
    ae_feature_npz: str,
    handcrafted_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    
    hand_df = load_labeled_dataframe(labeled_paths, handcrafted_features=handcrafted_features)
    ae_df = load_ae_feature_dataframe(ae_feature_npz)

    if len(hand_df) != len(ae_df):
        raise ValueError(f"Row count mismatch: handcrafted={len(hand_df)}, ae={len(ae_df)}")

    if not np.array_equal(hand_df["y_binary"].to_numpy(), ae_df["y_binary"].to_numpy()):
        raise ValueError("Label mismatch between handcrafted and AE feature files.")
    if not np.array_equal(hand_df["group_id"].to_numpy(), ae_df["group_id"].to_numpy()):
        raise ValueError("Group mismatch between handcrafted and AE feature files.")
    if not np.array_equal(hand_df["image_name"].to_numpy(), ae_df["image_name"].to_numpy()):
        raise ValueError("Image-name mismatch between handcrafted and AE feature files.")

    ae_cols = [c for c in ae_df.columns if c.startswith("ae_")]
    merged = pd.concat(
        [hand_df.reset_index(drop=True), ae_df[ae_cols].reset_index(drop=True)],
        axis=1,
    )

    if handcrafted_features is None:
        handcrafted_features = HANDCRAFTED_DEFAULT
    handcrafted_cols = list(handcrafted_features)
    combined_cols = handcrafted_cols + ae_cols

    feature_sets = {
        "handcrafted": handcrafted_cols,
        "ae_only": ae_cols,
        "combined": combined_cols,
    }
    return merged, feature_sets


def load_unlabeled_dataframe(
    unlabeled_paths: Sequence[str],
    handcrafted_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
   
    if handcrafted_features is None:
        handcrafted_features = HANDCRAFTED_DEFAULT

    rows: List[pd.DataFrame] = []
    for group_id, path in enumerate(unlabeled_paths):
        arr = _load_npz_array(path)
        if arr.shape[1] not in [10, 11]:
            raise ValueError(f"Unexpected column count in {path}: {arr.shape[1]}")
        if arr.shape[1] == 11:
            arr = arr[:, :-1]

        df = pd.DataFrame({
            "y_coord": arr[:, FEATURE_COLUMNS["y"]].astype(int),
            "x_coord": arr[:, FEATURE_COLUMNS["x"]].astype(int),
            "image_name": os.path.basename(path),
            "group_id": group_id,
        })
        for feat in handcrafted_features:
            if feat not in FEATURE_COLUMNS:
                raise KeyError(f"Unknown handcrafted feature: {feat}")
            df[feat] = arr[:, FEATURE_COLUMNS[feat]].astype(np.float32)
        rows.append(df.reset_index(drop=True))

    return pd.concat(rows, axis=0, ignore_index=True)


def load_unlabeled_ae_feature_dataframe(ae_feature_npz: str) -> pd.DataFrame:
    
    data = np.load(ae_feature_npz, allow_pickle=True)
    X = data["X"]
    groups = data["groups"].astype(int)
    image_names = data["image_names"]

    df = pd.DataFrame(X, columns=[f"ae_{i}" for i in range(X.shape[1])])
    df["group_id"] = groups
    df["image_name"] = [str(image_names[g]) for g in groups]

    if "y" in data.files:
        df["y_binary"] = data["y"].astype(int)

    return df


def assemble_unlabeled_feature_sets(
    unlabeled_paths: Sequence[str],
    handcrafted_features: Optional[Sequence[str]] = None,
    ae_feature_npz: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
   
    hand_df = load_unlabeled_dataframe(
        unlabeled_paths,
        handcrafted_features=handcrafted_features,
    )

    if handcrafted_features is None:
        handcrafted_features = HANDCRAFTED_DEFAULT
    handcrafted_cols = list(handcrafted_features)

    feature_sets: Dict[str, List[str]] = {
        "handcrafted": handcrafted_cols,
    }

    if ae_feature_npz is None:
        return hand_df, feature_sets

    ae_df = load_unlabeled_ae_feature_dataframe(ae_feature_npz)
    if len(hand_df) != len(ae_df):
        raise ValueError(f"Row count mismatch: handcrafted={len(hand_df)}, ae={len(ae_df)}")
    if not np.array_equal(hand_df["group_id"].to_numpy(), ae_df["group_id"].to_numpy()):
        raise ValueError("Group mismatch between unlabeled handcrafted and AE features.")
    if not np.array_equal(hand_df["image_name"].to_numpy(), ae_df["image_name"].to_numpy()):
        raise ValueError("Image-name mismatch between unlabeled handcrafted and AE features.")

    ae_cols = [c for c in ae_df.columns if c.startswith("ae_")]
    merged = pd.concat(
        [hand_df.reset_index(drop=True), ae_df[ae_cols].reset_index(drop=True)],
        axis=1,
    )

    feature_sets["ae_only"] = ae_cols
    feature_sets["combined"] = handcrafted_cols + ae_cols
    return merged, feature_sets