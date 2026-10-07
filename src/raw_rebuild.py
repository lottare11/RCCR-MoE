from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from data import AlignedModalities, SERS_MAX_WAVENUMBER, SERS_MIN_WAVENUMBER


@dataclass
class FoldSelection:
    tran_indices: np.ndarray
    meta_indices: np.ndarray
    sers_indices: np.ndarray
    tran_candidate_count: int
    meta_candidate_count: int


@dataclass
class RawModalities:
    sers: np.ndarray
    tran_fpkm: np.ndarray
    tran_count: np.ndarray
    meta: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    sers_names: np.ndarray
    tran_names: np.ndarray
    meta_names: np.ndarray


def load_raw_modalities(root: str | Path) -> RawModalities:
    root = Path(root)
    subjects = pd.read_csv(root / "canonical_subjects.csv")
    sers = np.load(root / "sers_mean3_139x1934.npy").astype(np.float32)
    tran_fpkm = np.load(root / "transcript_fpkm_139x62354.npy", mmap_mode="r")
    tran_count = np.load(root / "transcript_count_139x62354.npy", mmap_mode="r")
    meta = np.load(root / "metabolomics_full_139x2388.npy", mmap_mode="r")
    sers_names = np.loadtxt(root / "sers_wavenumbers.csv", delimiter=",").astype(np.float32)
    tran_info = pd.read_csv(root / "transcript_feature_metadata.csv", low_memory=False)
    meta_info = pd.read_csv(root / "metabolomics_feature_metadata.csv", low_memory=False)
    tran_names = tran_info["gene_name"].fillna(tran_info["gene_id"]).astype(str).to_numpy(object)
    # Preserve unique compound IDs including mode suffixes already present in the vendor IDs.
    meta_names = meta_info["Compound_ID"].astype(str).to_numpy(object)
    if not (len(subjects) == sers.shape[0] == tran_fpkm.shape[0] == tran_count.shape[0] == meta.shape[0]):
        raise ValueError("Raw modality sample counts are not aligned")
    return RawModalities(
        sers=sers,
        tran_fpkm=tran_fpkm,
        tran_count=tran_count,
        meta=meta,
        labels=subjects["label"].to_numpy(np.int64),
        ids=subjects["ID"].to_numpy(np.int64),
        sers_names=sers_names,
        tran_names=tran_names,
        meta_names=meta_names,
    )


def _rank_f_classif(x: np.ndarray, y: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    if len(indices) <= k:
        return indices.astype(np.int64)
    scores, _ = f_classif(x[:, indices], y)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
    order = np.lexsort((indices, scores))
    return indices[order[-k:]].astype(np.int64)


def fit_fold_selection(
    raw: RawModalities,
    train_idx: np.ndarray,
    tran_k: int = 512,
    meta_k: int = 128,
    tran_prevar_k: int = 5000,
    expressed_fpkm: float = 1.0,
    expressed_fraction: float = 0.20,
) -> FoldSelection:
    """Fit all data-dependent feature filtering using outer-fold training subjects only."""
    y = raw.labels[train_idx]
    tran_raw = np.asarray(raw.tran_fpkm[train_idx], dtype=np.float32)
    tran_log = np.log1p(tran_raw)
    prevalence = (tran_raw > expressed_fpkm).mean(axis=0)
    tran_candidates = np.where(prevalence >= expressed_fraction)[0]
    if len(tran_candidates) == 0:
        raise ValueError("No transcript features passed train-only prevalence filter")
    if len(tran_candidates) > tran_prevar_k:
        var = tran_log[:, tran_candidates].var(axis=0)
        order = np.lexsort((tran_candidates, var))
        tran_candidates = tran_candidates[order[-tran_prevar_k:]]
    tran_selected = _rank_f_classif(tran_log, y, tran_candidates, tran_k)

    meta_raw = np.asarray(raw.meta[train_idx], dtype=np.float32)
    meta_log = np.log1p(np.clip(meta_raw, a_min=0.0, a_max=None))
    meta_var = meta_log.var(axis=0)
    meta_candidates = np.where(meta_var > 1e-12)[0]
    if len(meta_candidates) == 0:
        raise ValueError("No metabolomics features passed train-only variance filter")
    meta_selected = _rank_f_classif(meta_log, y, meta_candidates, meta_k)

    sers_selected = np.where((raw.sers_names >= SERS_MIN_WAVENUMBER) & (raw.sers_names <= SERS_MAX_WAVENUMBER))[0].astype(np.int64)
    if len(sers_selected) == 0:
        raise ValueError("No SERS features remain in required wavenumber range")
    return FoldSelection(
        tran_indices=tran_selected,
        meta_indices=meta_selected,
        sers_indices=sers_selected,
        tran_candidate_count=int(len(tran_candidates)),
        meta_candidate_count=int(len(meta_candidates)),
    )


def materialize_selected(raw: RawModalities, idx: np.ndarray, selection: FoldSelection) -> AlignedModalities:
    sers = raw.sers[idx][:, selection.sers_indices].astype(np.float32)
    tran = np.log1p(np.asarray(raw.tran_fpkm[idx][:, selection.tran_indices], dtype=np.float32))
    meta = np.log1p(np.clip(np.asarray(raw.meta[idx][:, selection.meta_indices], dtype=np.float32), a_min=0.0, a_max=None))
    return AlignedModalities(
        sers=sers,
        tran=tran,
        meta=meta,
        labels=raw.labels[idx].copy(),
        ids=raw.ids[idx].copy(),
        sers_names=raw.sers_names[selection.sers_indices].astype(object),
        tran_names=raw.tran_names[selection.tran_indices],
        meta_names=raw.meta_names[selection.meta_indices],
    )


def fit_fold_scalers(train: AlignedModalities) -> Dict[str, StandardScaler]:
    scalers = {
        "sers": StandardScaler().fit(train.sers),
        "tran": StandardScaler().fit(train.tran),
        "meta": StandardScaler().fit(train.meta),
    }
    return scalers


def apply_fold_scalers(data: AlignedModalities, scalers: Dict[str, StandardScaler]) -> AlignedModalities:
    return AlignedModalities(
        sers=scalers["sers"].transform(data.sers).astype(np.float32),
        tran=scalers["tran"].transform(data.tran).astype(np.float32),
        meta=scalers["meta"].transform(data.meta).astype(np.float32),
        labels=data.labels,
        ids=data.ids,
        sers_names=data.sers_names,
        tran_names=data.tran_names,
        meta_names=data.meta_names,
    )
