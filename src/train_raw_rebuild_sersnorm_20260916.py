from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from data import TripleTensorDataset, build_feature_group_spec
from raw_rebuild import (
    apply_fold_scalers,
    fit_fold_scalers,
    fit_fold_selection,
    load_raw_modalities,
    materialize_selected,
)
from deep_baselines import build_deep_baseline
from model import build_tricor_moe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tn / max(tn + fp, 1))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    brier = float(np.mean((y_prob - y_true.astype(np.float32)) ** 2))
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
        "brier": brier,
        "threshold": float(threshold),
    }


def balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    return float(0.5 * (recall_score(y_true, y_pred, zero_division=0) + specificity_score(y_true, y_pred)))


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1") -> float:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= thr).astype(int)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = 0.5 * (recall_score(y_true, y_pred, zero_division=0) + specificity_score(y_true, y_pred))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def fixed_mask(mask: Tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(mask, dtype=torch.float32).view(1, 3)


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device, modality_mask=None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for sers, tran, meta, y in loader:
            sers = sers.to(device)
            tran = tran.to(device)
            meta = meta.to(device)
            mask = None
            if modality_mask is not None:
                mask = modality_mask.to(device).expand(sers.size(0), -1)
            out = model(sers, tran, meta, modality_mask=mask)
            prob = F.softmax(out["logits"], dim=-1)[:, 1].detach().cpu().numpy()
            ys.append(y.numpy())
            probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5, modality_mask=None) -> Dict[str, float]:
    y_true, y_prob = predict_probs(model, loader, device, modality_mask=modality_mask)
    return binary_metrics(y_true, y_prob, threshold=threshold)


def role_token_aux_loss(out: Dict[str, torch.Tensor], args) -> torch.Tensor:
    if "shared_sers" not in out:
        return out["logits"].new_zeros(())
    align_loss = (
        F.mse_loss(out["shared_sers"], out["shared_tran"])
        + F.mse_loss(out["shared_sers"], out["shared_meta"])
        + F.mse_loss(out["shared_tran"], out["shared_meta"])
        + F.mse_loss(out["shared_sers"], out["consensus_src"])
        + F.mse_loss(out["shared_tran"], out["consensus_src"])
        + F.mse_loss(out["shared_meta"], out["consensus_src"])
    ) / 6.0
    consensus_loss = F.mse_loss(out["consensus_token"], out["consensus_src"])
    separation_loss = (
        F.cosine_similarity(out["shared_sers"], out["private_sers"], dim=-1).pow(2).mean()
        + F.cosine_similarity(out["shared_tran"], out["private_tran"], dim=-1).pow(2).mean()
        + F.cosine_similarity(out["shared_meta"], out["private_meta"], dim=-1).pow(2).mean()
    ) / 3.0
    gate_entropy = out["logits"].new_zeros(())
    if "gate" in out:
        gate = out["gate"].clamp_min(1e-8)
        gate_entropy = -(gate * gate.log()).sum(dim=-1).mean()
    return args.align_weight * align_loss + args.consensus_weight * consensus_loss + args.separation_weight * separation_loss - args.entropy_weight * gate_entropy


def role_token_expert_loss(out: Dict[str, torch.Tensor], y: torch.Tensor, ce_weight: torch.Tensor) -> torch.Tensor:
    if "expert_logits" not in out:
        return y.new_zeros((), dtype=torch.float32)
    expert_logits = out["expert_logits"]
    losses = [F.cross_entropy(expert_logits[:, k, :], y, weight=ce_weight) for k in range(expert_logits.size(1))]
    return torch.stack(losses).mean()


def apply_sers_sample_norm(data, mode: str):
    if mode == "none":
        return data
    x = data.sers.astype(np.float32, copy=True)
    if mode == "snv":
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        x = (x - mu) / np.maximum(sd, 1e-6)
    elif mode == "l2":
        norm = np.sqrt((x * x).sum(axis=1, keepdims=True))
        x = x / np.maximum(norm, 1e-6)
    else:
        raise ValueError(f"Unknown sers sample norm: {mode}")
    data.sers = x.astype(np.float32)
    return data


def train_one_model(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_raw_modalities(args.raw_data_dir)
    out_dir = Path(args.out_dir) / args.model_name / f"seed_{args.seed}"
    train_mask_tuple = tuple(int(c) for c in args.train_mask)
    if len(train_mask_tuple) != 3 or sum(train_mask_tuple) < 1:
        raise ValueError("--train-mask must be 3 binary digits, e.g. 110")
    train_mask_fixed = fixed_mask(train_mask_tuple)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results: List[Dict[str, float]] = []
    detailed: List[Dict[str, Dict[str, float]]] = []
    fold_metadata: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    outer_splits = list(skf.split(np.zeros_like(data.labels), data.labels))
    for fold in range(args.n_splits):
        train_val_idx, test_idx = outer_splits[fold]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            random_state=args.seed + fold,
            stratify=data.labels[train_val_idx],
        )
        selection = fit_fold_selection(
            data,
            train_idx,
            tran_k=args.tran_k,
            meta_k=args.meta_k,
            tran_prevar_k=args.tran_prevar_k,
            expressed_fpkm=args.expressed_fpkm,
            expressed_fraction=args.expressed_fraction,
        )
        train_data = materialize_selected(data, train_idx, selection)
        val_data = materialize_selected(data, val_idx, selection)
        test_data = materialize_selected(data, test_idx, selection)
        train_data = apply_sers_sample_norm(train_data, args.sers_sample_norm)
        val_data = apply_sers_sample_norm(val_data, args.sers_sample_norm)
        test_data = apply_sers_sample_norm(test_data, args.sers_sample_norm)
        group_spec = build_feature_group_spec(train_data, seed=args.seed + fold, sers_groups=args.sers_groups, tran_groups=args.tran_groups, meta_groups=args.meta_groups)

        scalers = fit_fold_scalers(train_data)
        train_data = apply_fold_scalers(train_data, scalers)
        val_data = apply_fold_scalers(val_data, scalers)
        test_data = apply_fold_scalers(test_data, scalers)

        train_loader = DataLoader(TripleTensorDataset(train_data.sers, train_data.tran, train_data.meta, train_data.labels), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TripleTensorDataset(val_data.sers, val_data.tran, val_data.meta, val_data.labels), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TripleTensorDataset(test_data.sers, test_data.tran, test_data.meta, test_data.labels), batch_size=args.batch_size, shuffle=False)

        class_counts = np.bincount(train_data.labels, minlength=2).astype(np.float32)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.mean()
        ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

        if args.model_name == "tricor_moe":
            model = build_tricor_moe(
                group_spec,
                d_model=args.d_model,
                dropout=args.dropout,
                use_consensus_token=not args.no_consensus_token,
                use_conflict_token=not args.no_conflict_token,
                use_shared_private=not args.no_shared_private,
                use_moe_routing=not args.no_moe_routing,
                use_refinement_head=not args.no_refinement_head,
            )
        else:
            model = build_deep_baseline(
                args.model_name,
                train_data.sers.shape[1],
                train_data.tran.shape[1],
                train_data.meta.shape[1],
                d_model=args.d_model,
                dropout=args.dropout,
            )
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_auc = -1.0
        best_state = None
        patience = 0

        for _ in range(args.epochs):
            model.train()
            for sers, tran, meta, y in train_loader:
                sers = sers.to(device)
                tran = tran.to(device)
                meta = meta.to(device)
                y = y.to(device)

                full_mask = train_mask_fixed.to(device).expand(sers.size(0), -1).clone()
                mask = full_mask.clone()
                if args.modality_dropout > 0:
                    drop = (torch.rand(sers.size(0), 3, device=device) < args.modality_dropout) & (full_mask > 0)
                    mask = mask.masked_fill(drop, 0.0)
                    empty = mask.sum(dim=1) == 0
                    if empty.any():
                        first_alive = int(next(i for i,v in enumerate(train_mask_tuple) if v))
                        mask[empty, first_alive] = 1.0
                    if args.keep_sers_alive and train_mask_tuple[0] == 1:
                        mask[:, 0] = 1.0

                out = model(sers, tran, meta, modality_mask=full_mask)
                loss = F.cross_entropy(out["logits"], y, weight=ce_weight)
                aux_loss = role_token_expert_loss(out, y, ce_weight) + out.get("aux_loss", out["logits"].new_zeros(()))
                loss = loss + args.aux_weight * aux_loss

                if args.model_name == "tricor_moe" or args.consistency_weight > 0:
                    masked_out = model(sers, tran, meta, modality_mask=mask)
                    masked_loss = F.cross_entropy(masked_out["logits"], y, weight=ce_weight)
                    consistency_loss = F.kl_div(F.log_softmax(masked_out["logits"], dim=-1), F.softmax(out["logits"].detach(), dim=-1), reduction="batchmean")
                    loss = loss + 0.5 * masked_loss + args.consistency_weight * consistency_loss

                if args.model_name == "tricor_moe":
                    loss = loss + role_token_aux_loss(out, args)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            val_y, val_prob = predict_probs(model, val_loader, device, train_mask_fixed)
            val_auc = roc_auc_score(val_y, val_prob)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(
                {"state_dict": best_state, "args": vars(args), "fold": fold, "group_spec": group_spec.__dict__},
                out_dir / f"fold_{fold}_best.pt",
            )

        val_y, val_prob = predict_probs(model, val_loader, device, train_mask_fixed)
        threshold = select_threshold(val_y, val_prob, objective=args.threshold_objective)
        val_metrics = binary_metrics(val_y, val_prob, threshold)
        val_metrics["balanced_acc"] = balanced_accuracy(val_y, val_prob, threshold)
        test_y, test_prob = predict_probs(model, test_loader, device, train_mask_fixed)
        test_pred = (test_prob >= threshold).astype(int)
        for subject_id, y_true, prob, pred in zip(test_data.ids, test_y, test_prob, test_pred):
            oof_rows.append(
                {
                    "subject_id": int(subject_id),
                    "fold": int(fold),
                    "true_label": int(y_true),
                    "predicted_probability": float(prob),
                    "predicted_label": int(pred),
                    "threshold": float(threshold),
                }
            )

        cases = {
            "full": binary_metrics(test_y, test_prob, threshold),
            "sers_only": evaluate_model(model, test_loader, device, threshold, fixed_mask((1, 0, 0))),
            "tran_only": evaluate_model(model, test_loader, device, threshold, fixed_mask((0, 1, 0))),
            "meta_only": evaluate_model(model, test_loader, device, threshold, fixed_mask((0, 0, 1))),
            "drop_sers": evaluate_model(model, test_loader, device, threshold, fixed_mask((0, 1, 1))),
            "drop_tran": evaluate_model(model, test_loader, device, threshold, fixed_mask((1, 0, 1))),
            "drop_meta": evaluate_model(model, test_loader, device, threshold, fixed_mask((1, 1, 0))),
        }
        cases["full"]["fold"] = fold
        fold_results.append(cases["full"])
        detailed.append(cases)
        fold_metadata.append(
            {
                "fold": fold,
                "threshold": float(threshold),
                "best_val_auc": float(best_val_auc),
                "val_metrics": val_metrics,
                "train_ids": train_data.ids.astype(int).tolist(),
                "val_ids": val_data.ids.astype(int).tolist(),
                "test_ids": test_data.ids.astype(int).tolist(),
                "train_size": int(len(train_data.ids)),
                "val_size": int(len(val_data.ids)),
                "test_size": int(len(test_data.ids)),
                "train_pos": int(train_data.labels.sum()),
                "val_pos": int(val_data.labels.sum()),
                "test_pos": int(test_data.labels.sum()),
                "tran_selected_count": int(len(selection.tran_indices)),
                "meta_selected_count": int(len(selection.meta_indices)),
                "sers_selected_count": int(len(selection.sers_indices)),
                "tran_candidate_count": int(selection.tran_candidate_count),
                "meta_candidate_count": int(selection.meta_candidate_count),
                "tran_selected_names": [str(x) for x in train_data.tran_names.tolist()],
                "meta_selected_names": [str(x) for x in train_data.meta_names.tolist()],
            }
        )
        print(f"[{args.model_name}][fold {fold}] " + json.dumps(cases["full"], ensure_ascii=False, sort_keys=True))

    metric_keys = [k for k in fold_results[0].keys() if k != "fold"]
    summary = {}
    for k in metric_keys:
        vals = np.array([m[k] for m in fold_results], dtype=np.float32)
        summary[f"full_{k}_mean"] = float(vals.mean())
        summary[f"full_{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    for case in ["sers_only", "tran_only", "meta_only", "drop_sers", "drop_tran", "drop_meta"]:
        case_rows = [m[case] for m in detailed]
        for k in metric_keys:
            vals = np.array([r[k] for r in case_rows], dtype=np.float32)
            summary[f"{case}_{k}_mean"] = float(vals.mean())
            summary[f"{case}_{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fold_metrics.json").write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fold_metadata.json").write_text(json.dumps(fold_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    oof_path = out_dir / "oof_predictions.csv"
    with oof_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "fold",
                "true_label",
                "predicted_probability",
                "predicted_label",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(oof_rows, key=lambda row: (int(row["fold"]), int(row["subject_id"]))))
    return {"model_name": args.model_name, "summary": summary, "out_dir": str(out_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, choices=["tricor_moe", "early_mlp", "late_mlp", "gated_fusion", "tensor_fusion", "cross_attention_fusion"])
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--raw-data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "raw_rebuild_20260916"))
    parser.add_argument("--train-mask", type=str, default="111")
    parser.add_argument("--sers-sample-norm", choices=["none","snv","l2"], default="none")
    parser.add_argument("--tran-k", type=int, default=512)
    parser.add_argument("--meta-k", type=int, default=128)
    parser.add_argument("--tran-prevar-k", type=int, default=5000)
    parser.add_argument("--expressed-fpkm", type=float, default=1.0)
    parser.add_argument("--expressed-fraction", type=float, default=0.20)
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "seed1002"))
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--norm-mode", type=str, default="zscore", choices=["zscore", "l2"])
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--consensus-weight", type=float, default=0.15)
    parser.add_argument("--separation-weight", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--threshold-objective", type=str, default="balanced_acc", choices=["f1", "balanced_acc"])
    parser.add_argument("--sers-groups", type=int, default=16)
    parser.add_argument("--tran-groups", type=int, default=24)
    parser.add_argument("--meta-groups", type=int, default=8)
    parser.add_argument("--keep-sers-alive", action="store_true")
    parser.add_argument("--no-consensus-token", action="store_true")
    parser.add_argument("--no-conflict-token", action="store_true")
    parser.add_argument("--no-shared-private", action="store_true")
    parser.add_argument("--no-moe-routing", action="store_true")
    parser.add_argument("--no-refinement-head", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_one_model(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
