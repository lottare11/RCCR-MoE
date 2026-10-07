from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import Dataset


@dataclass
class AlignedModalities:
    sers: np.ndarray
    tran: np.ndarray
    meta: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    sers_names: np.ndarray
    tran_names: np.ndarray
    meta_names: np.ndarray


SERS_MIN_WAVENUMBER = 68.0
SERS_MAX_WAVENUMBER = 4000.0


@dataclass
class FeatureGroupSpec:
    sers_groups: List[List[int]]
    tran_groups: List[List[int]]
    meta_groups: List[List[int]]
    sers_group_labels: List[str]
    tran_group_labels: List[str]
    meta_group_labels: List[str]


class TripleTensorDataset(Dataset):
    def __init__(self, sers: np.ndarray, tran: np.ndarray, meta: np.ndarray, labels: np.ndarray) -> None:
        self.sers = torch.tensor(sers, dtype=torch.float32)
        self.tran = torch.tensor(tran, dtype=torch.float32)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sers[idx], self.tran[idx], self.meta[idx], self.labels[idx]


def _read_table(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.sort_values(by=df.columns[1], ascending=True, ignore_index=True)
    values = df.values
    labels = values[:, 0].astype(np.int64)
    ids = values[:, 1].astype(np.int64)
    feats = values[:, 2:].astype(np.float32)
    feat_names = np.array([str(c) for c in df.columns[2:]], dtype=object)
    return feats, labels, ids, feat_names


def load_aligned_modalities(data_dir: str | Path) -> AlignedModalities:
    data_dir = Path(data_dir)
    sers_x, sers_y, sers_id, sers_names = _read_table(data_dir / "A_B_sers_sorted.xlsx")
    tran_x, tran_y, tran_id, tran_names = _read_table(data_dir / "A_B_tran_sorted.xlsx")
    meta_x, meta_y, meta_id, meta_names = _read_table(data_dir / "A_B_meta_sorted.csv")
    tran_y = 1 - tran_y

    sers_wavenumbers = np.array([float(v) for v in sers_names], dtype=np.float32)
    sers_keep = (sers_wavenumbers >= SERS_MIN_WAVENUMBER) & (sers_wavenumbers <= SERS_MAX_WAVENUMBER)
    if not np.any(sers_keep):
        raise ValueError("No SERS features remain after applying the 68-4000 cm^-1 range.")
    sers_x = sers_x[:, sers_keep]
    sers_names = sers_names[sers_keep]

    sers_map = {int(v): i for i, v in enumerate(sers_id)}
    tran_map = {int(v): i for i, v in enumerate(tran_id)}
    meta_map = {int(v): i for i, v in enumerate(meta_id)}
    common_ids = sorted(set(sers_map) & set(tran_map) & set(meta_map))

    sers_idx = np.array([sers_map[i] for i in common_ids], dtype=np.int64)
    tran_idx = np.array([tran_map[i] for i in common_ids], dtype=np.int64)
    meta_idx = np.array([meta_map[i] for i in common_ids], dtype=np.int64)
    labels = sers_y[sers_idx].astype(np.int64)

    return AlignedModalities(
        sers=sers_x[sers_idx],
        tran=tran_x[tran_idx],
        meta=meta_x[meta_idx],
        labels=labels,
        ids=np.array(common_ids, dtype=np.int64),
        sers_names=sers_names,
        tran_names=tran_names,
        meta_names=meta_names,
    )


def subset(data: AlignedModalities, idx: np.ndarray) -> AlignedModalities:
    return AlignedModalities(
        sers=data.sers[idx],
        tran=data.tran[idx],
        meta=data.meta[idx],
        labels=data.labels[idx],
        ids=data.ids[idx],
        sers_names=data.sers_names,
        tran_names=data.tran_names,
        meta_names=data.meta_names,
    )


def make_split(data: AlignedModalities, seed: int, fold: int, n_splits: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros_like(data.labels), data.labels))
    train_val_idx, test_idx = splits[fold]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=seed + fold,
        stratify=data.labels[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def fit_transform_modalities(train: AlignedModalities, norm_mode: str = "zscore"):
    if norm_mode == "zscore":
        scaler_sers = StandardScaler()
        scaler_tran = StandardScaler()
        scaler_meta = StandardScaler()
    elif norm_mode == "l2":
        scaler_sers = Normalizer(norm="l2")
        scaler_tran = Normalizer(norm="l2")
        scaler_meta = Normalizer(norm="l2")
    else:
        raise ValueError(f"Unsupported norm_mode: {norm_mode}")
    train_sers = scaler_sers.fit_transform(train.sers)
    train_tran = scaler_tran.fit_transform(train.tran)
    train_meta = scaler_meta.fit_transform(train.meta)
    scalers = {"sers": scaler_sers, "tran": scaler_tran, "meta": scaler_meta}
    return (
        AlignedModalities(train_sers, train_tran, train_meta, train.labels, train.ids, train.sers_names, train.tran_names, train.meta_names),
        scalers,
    )


def transform_modalities(data: AlignedModalities, scalers: Dict[str, object]) -> AlignedModalities:
    return AlignedModalities(
        sers=scalers["sers"].transform(data.sers),
        tran=scalers["tran"].transform(data.tran),
        meta=scalers["meta"].transform(data.meta),
        labels=data.labels,
        ids=data.ids,
        sers_names=data.sers_names,
        tran_names=data.tran_names,
        meta_names=data.meta_names,
    )


def _make_equal_size_chunks(indices: np.ndarray, n_groups: int) -> List[np.ndarray]:
    return [np.asarray(c, dtype=np.int64) for c in np.array_split(indices, n_groups) if len(c) > 0]


def build_sers_groups(feature_names: np.ndarray, n_groups: int = 16) -> Tuple[List[List[int]], List[str]]:
    wavenumbers = np.array([float(x) for x in feature_names], dtype=np.float32)
    if float(wavenumbers.min()) < SERS_MIN_WAVENUMBER or float(wavenumbers.max()) > SERS_MAX_WAVENUMBER:
        raise ValueError("SERS grouping received features outside the required 68-4000 cm^-1 range.")
    order = np.argsort(wavenumbers)
    bins = _make_equal_size_chunks(order, n_groups)
    groups: List[List[int]] = []
    labels: List[str] = []
    for g in bins:
        groups.append(g.tolist())
        labels.append(f"{wavenumbers[g].min():.0f}-{wavenumbers[g].max():.0f}")
    return groups, labels


def build_module_groups(x: np.ndarray, n_groups: int, feature_names: np.ndarray, random_state: int = 0) -> Tuple[List[List[int]], List[str]]:
    if x.shape[1] <= n_groups:
        groups = [[int(i)] for i in range(x.shape[1])]
        labels = [str(feature_names[i]) for i in range(x.shape[1])]
        return groups, labels
    x_feat = x.T.astype(np.float32)
    x_feat = x_feat - x_feat.mean(axis=1, keepdims=True)
    x_feat = x_feat / (x_feat.std(axis=1, keepdims=True) + 1e-6)
    x_feat = np.nan_to_num(x_feat)
    x_feat = x_feat / (np.linalg.norm(x_feat, axis=1, keepdims=True) + 1e-6)
    km = KMeans(n_clusters=n_groups, random_state=random_state, n_init=20)
    labels_raw = km.fit_predict(x_feat)
    groups: List[List[int]] = []
    labels: List[str] = []
    for cluster_id in range(n_groups):
        idx = np.where(labels_raw == cluster_id)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(idx)]
        groups.append(idx.tolist())
        labels.append(" | ".join(str(feature_names[i]) for i in idx[: min(3, len(idx))]))
    return groups, labels


def build_feature_group_spec(train: AlignedModalities, seed: int = 0, sers_groups: int = 16, tran_groups: int = 24, meta_groups: int = 8) -> FeatureGroupSpec:
    sers_idx_groups, sers_labels = build_sers_groups(train.sers_names, n_groups=sers_groups)
    tran_idx_groups, tran_labels = build_module_groups(train.tran, n_groups=tran_groups, feature_names=train.tran_names, random_state=seed)
    meta_idx_groups, meta_labels = build_module_groups(train.meta, n_groups=meta_groups, feature_names=train.meta_names, random_state=seed + 17)
    return FeatureGroupSpec(
        sers_groups=sers_idx_groups,
        tran_groups=tran_idx_groups,
        meta_groups=meta_idx_groups,
        sers_group_labels=sers_labels,
        tran_group_labels=tran_labels,
        meta_group_labels=meta_labels,
    )
