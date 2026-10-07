from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from data import TripleTensorDataset, build_feature_group_spec
from raw_rebuild import load_raw_modalities, fit_fold_selection, materialize_selected, fit_fold_scalers, apply_fold_scalers
from train_raw_rebuild_sersnorm_20260916 import apply_sers_sample_norm
from model import ROLE_NAMES, build_tricor_moe
from train import set_seed


def margin(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    args = ckpt["args"]
    group_spec = SimpleNamespace(**ckpt["group_spec"])
    model = build_tricor_moe(
        group_spec,
        d_model=int(args["d_model"]),
        dropout=float(args["dropout"]),
        use_consensus_token=not bool(args.get("no_consensus_token", False)),
        use_conflict_token=not bool(args.get("no_conflict_token", False)),
        use_shared_private=not bool(args.get("no_shared_private", False)),
        use_moe_routing=not bool(args.get("no_moe_routing", False)),
        use_refinement_head=not bool(args.get("no_refinement_head", False)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_outputs(model, data, device):
    loader = DataLoader(TripleTensorDataset(data.sers, data.tran, data.meta, data.labels), batch_size=len(data.labels), shuffle=False)
    with torch.no_grad():
        sers, tran, meta, y = next(iter(loader))
        out = model(sers.to(device), tran.to(device), meta.to(device))
        prob = F.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
    return out, y.numpy(), prob


def grad_input(model, data, modality: str, device: torch.device) -> np.ndarray:
    sers = torch.tensor(data.sers, dtype=torch.float32, device=device, requires_grad=(modality == "sers"))
    tran = torch.tensor(data.tran, dtype=torch.float32, device=device, requires_grad=(modality == "transcriptome"))
    meta = torch.tensor(data.meta, dtype=torch.float32, device=device, requires_grad=(modality == "metabolome"))
    out = model(sers, tran, meta)
    target = margin(out["logits"]).sum()
    model.zero_grad(set_to_none=True)
    target.backward()
    x = {"sers": sers, "transcriptome": tran, "metabolome": meta}[modality]
    return (x.grad * x).detach().cpu().numpy()


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def occlude_predict(model, data, modality: str, idx: np.ndarray, device: torch.device) -> np.ndarray:
    sers, tran, meta = np.array(data.sers, copy=True), np.array(data.tran, copy=True), np.array(data.meta, copy=True)
    if modality == "sers":
        sers[:, idx] = 0.0
    elif modality == "transcriptome":
        tran[:, idx] = 0.0
    else:
        meta[:, idx] = 0.0
    tmp = SimpleNamespace(sers=sers, tran=tran, meta=meta, labels=data.labels)
    _, _, prob = predict_outputs(model, tmp, device)
    return prob


def summarize_distribution(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        row = {group_col: group, "n": len(sub)}
        for col in value_cols:
            q1, q3 = sub[col].quantile([0.25, 0.75])
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_sd"] = sub[col].std(ddof=1)
            row[f"{col}_median"] = sub[col].median()
            row[f"{col}_iqr"] = q3 - q1
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out OOF TriCoR-MoE explanations.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw_rebuild_20260916")
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/model_selection/config_B_full/tricor_moe/seed_1002")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/interpretability")
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--random-repeats", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_modalities(args.raw_data_dir)
    metadata = json.loads((args.run_dir / "fold_metadata.json").read_text())
    ckpt0 = torch.load(args.run_dir / "fold_0_best.pt", map_location="cpu")
    train_args = ckpt0["args"]
    if int(train_args.get("seed", args.seed)) != int(args.seed):
        raise ValueError("Explanation seed does not match trained model seed")
    outer = list(StratifiedKFold(n_splits=int(train_args["n_splits"]), shuffle=True, random_state=args.seed).split(np.zeros_like(raw.labels), raw.labels))
    contrib_rows, routing_rows, landscape_rows = [], [], []
    sers_rows, tran_rows, meta_rows = [], [], []
    fidelity_rows, random_rows = [], []
    band_rows = []

    for fold in range(int(train_args["n_splits"])):
        train_val_idx, test_idx = outer[fold]
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=args.seed + fold, stratify=raw.labels[train_val_idx])
        selection = fit_fold_selection(raw, train_idx, tran_k=int(train_args["tran_k"]), meta_k=int(train_args["meta_k"]), tran_prevar_k=int(train_args["tran_prevar_k"]), expressed_fpkm=float(train_args["expressed_fpkm"]), expressed_fraction=float(train_args["expressed_fraction"]))
        train_data = materialize_selected(raw, train_idx, selection)
        val_data = materialize_selected(raw, val_idx, selection)
        test_data = materialize_selected(raw, test_idx, selection)
        train_data = apply_sers_sample_norm(train_data, train_args.get("sers_sample_norm", "none"))
        val_data = apply_sers_sample_norm(val_data, train_args.get("sers_sample_norm", "none"))
        test_data = apply_sers_sample_norm(test_data, train_args.get("sers_sample_norm", "none"))
        group_spec = build_feature_group_spec(train_data, seed=args.seed + fold, sers_groups=int(train_args["sers_groups"]), tran_groups=int(train_args["tran_groups"]), meta_groups=int(train_args["meta_groups"]))
        scalers = fit_fold_scalers(train_data)
        train_data = apply_fold_scalers(train_data, scalers)
        val_data = apply_fold_scalers(val_data, scalers)
        test_data = apply_fold_scalers(test_data, scalers)
        model = load_model(args.run_dir / f"fold_{fold}_best.pt", device)

        out, y, prob = predict_outputs(model, test_data, device)
        threshold = float(metadata[fold]["threshold"])
        pred = (prob >= threshold).astype(int)
        correct = pred == y
        logits = out["logits"].detach().cpu()
        expert_logits = out["expert_logits"].detach().cpu()
        role_logits = out["role_logits"].detach().cpu()
        global_logits = out["global_logits"].detach().cpu()
        refinement_logits = out["refinement_logits"].detach().cpu()
        gate = out["gate"].detach().cpu().numpy()
        pair_cons = out["pairwise_consensus"].detach().cpu().numpy()
        pair_conf = out["pairwise_conflict"].detach().cpu().numpy()
        expert_margins = (expert_logits[:, :, 1] - expert_logits[:, :, 0]).numpy()
        total_margin = (logits[:, 1] - logits[:, 0]).numpy()
        role_margin = (role_logits[:, 1] - role_logits[:, 0]).numpy()
        global_margin = (global_logits[:, 1] - global_logits[:, 0]).numpy()
        refinement_margin = (refinement_logits[:, 1] - refinement_logits[:, 0]).numpy()

        for i, subject_id in enumerate(test_data.ids):
            role_contrib = 0.5 * gate[i] * expert_margins[i]
            global_contrib = 0.5 * global_margin[i]
            refinement_contrib = refinement_margin[i]
            reconstructed = float(role_contrib.sum() + global_contrib + refinement_contrib)
            contrib_rows.append(
                {
                    "subject_id": int(subject_id),
                    "fold": fold,
                    "true_label": int(y[i]),
                    "predicted_probability": float(prob[i]),
                    "correct": bool(correct[i]),
                    "contrib_sers": float(role_contrib[0]),
                    "contrib_transcriptome": float(role_contrib[1]),
                    "contrib_metabolome": float(role_contrib[2]),
                    "contrib_consensus": float(role_contrib[3]),
                    "contrib_conflict": float(role_contrib[4]),
                    "contrib_fusion": float(role_contrib[5]),
                    "contrib_global": float(global_contrib),
                    "contrib_refinement": float(refinement_contrib),
                    "total_margin": float(total_margin[i]),
                    "reconstructed_margin": reconstructed,
                    "reconstruction_error": float(abs(total_margin[i] - reconstructed)),
                }
            )
            routing_rows.append(
                {
                    "subject_id": int(subject_id),
                    "fold": fold,
                    "label": int(y[i]),
                    "correct": bool(correct[i]),
                    "probability": float(prob[i]),
                    **{f"gate_{name}": float(gate[i, j]) for j, name in enumerate(ROLE_NAMES)},
                    "dominant_role": ROLE_NAMES[int(np.argmax(gate[i]))],
                }
            )
            landscape_rows.append(
                {
                    "subject_id": int(subject_id),
                    "fold": fold,
                    "label": int(y[i]),
                    "correct": bool(correct[i]),
                    "probability": float(prob[i]),
                    "prediction_confidence": float(max(prob[i], 1 - prob[i])),
                    "pairwise_consensus_mean": float(pair_cons[i].mean()),
                    "pairwise_consensus_sd": float(pair_cons[i].std(ddof=1)),
                    "pairwise_conflict_mean": float(pair_conf[i].mean()),
                    "expert_disagreement": float(expert_margins[i].std(ddof=1)),
                    "role_margin": float(role_margin[i]),
                }
            )

        attrs = {
            "sers": grad_input(model, test_data, "sers", device),
            "transcriptome": grad_input(model, test_data, "transcriptome", device),
            "metabolome": grad_input(model, test_data, "metabolome", device),
        }
        for i, subject_id in enumerate(test_data.ids):
            for j, name in enumerate(test_data.sers_names):
                sers_rows.append({"subject_id": int(subject_id), "fold": fold, "feature": str(name), "attribution": float(attrs["sers"][i, j]), "abs_attribution": float(abs(attrs["sers"][i, j]))})
            for j, name in enumerate(test_data.tran_names):
                tran_rows.append({"subject_id": int(subject_id), "fold": fold, "feature": str(name), "attribution": float(attrs["transcriptome"][i, j]), "abs_attribution": float(abs(attrs["transcriptome"][i, j]))})
            for j, name in enumerate(test_data.meta_names):
                meta_rows.append({"subject_id": int(subject_id), "fold": fold, "feature": str(name), "attribution": float(attrs["metabolome"][i, j]), "abs_attribution": float(abs(attrs["metabolome"][i, j]))})

        sers_mean = pd.DataFrame({"feature": [str(v) for v in test_data.sers_names], "mean_abs": np.abs(attrs["sers"]).mean(axis=0)})
        sers_lookup = dict(zip(sers_mean["feature"], sers_mean["mean_abs"]))
        fold_band_values = []
        for band_i, group in enumerate(group_spec.sers_groups):
            names = [str(test_data.sers_names[j]) for j in group]
            vals = np.array([sers_lookup[n] for n in names], dtype=float)
            fold_band_values.append((band_i, min(map(float, names)), max(map(float, names)), vals.mean(), vals.std(ddof=1)))
        ranks = {fold_band_values[idx][0]: rank + 1 for rank, idx in enumerate(np.argsort([-x[3] for x in fold_band_values]))}
        for band_i, lo, hi, mean_abs, sd_abs in fold_band_values:
            band_rows.append({"fold": fold, "band": band_i, "lower_cm1": lo, "upper_cm1": hi, "mean_abs_attribution": mean_abs, "SD": sd_abs, "rank_per_fold": ranks[band_i], "top5": ranks[band_i] <= 5})

        base_prob = prob
        base_auc = safe_auc(y, base_prob)
        base_ll = log_loss(y, base_prob, labels=[0, 1])
        base_brier = float(np.mean((base_prob - y) ** 2))
        for modality, val_names in [("sers", val_data.sers_names), ("transcriptome", val_data.tran_names), ("metabolome", val_data.meta_names)]:
            val_attr = np.abs(grad_input(model, val_data, modality, device)).mean(axis=0)
            ranking = np.argsort(-val_attr)
            n_features = len(ranking)
            for pct in [0.01, 0.05, 0.10, 0.20]:
                k = max(1, int(round(n_features * pct)))
                top_idx = ranking[:k]
                top_prob = occlude_predict(model, test_data, modality, top_idx, device)
                fidelity_rows.append({"fold": fold, "modality": modality, "percent": pct, "kind": "targeted", "n_features": k, "delta_log_loss": float(log_loss(y, top_prob, labels=[0, 1]) - base_ll), "delta_brier": float(np.mean((top_prob - y) ** 2) - base_brier), "delta_auc": safe_auc(y, top_prob) - base_auc, "delta_mean_positive_probability": float(top_prob.mean() - base_prob.mean()), "mean_abs_probability_change": float(np.mean(np.abs(top_prob - base_prob)))})
                for rep in range(args.random_repeats):
                    rand_idx = rng.choice(n_features, size=k, replace=False)
                    rand_prob = occlude_predict(model, test_data, modality, rand_idx, device)
                    random_rows.append({"fold": fold, "modality": modality, "percent": pct, "repeat": rep, "n_features": k, "delta_log_loss": float(log_loss(y, rand_prob, labels=[0, 1]) - base_ll), "delta_brier": float(np.mean((rand_prob - y) ** 2) - base_brier), "delta_auc": safe_auc(y, rand_prob) - base_auc, "delta_mean_positive_probability": float(rand_prob.mean() - base_prob.mean()), "mean_abs_probability_change": float(np.mean(np.abs(rand_prob - base_prob)))})

    contrib = pd.DataFrame(contrib_rows)
    routing = pd.DataFrame(routing_rows)
    landscape = pd.DataFrame(landscape_rows)
    sers_attr = pd.DataFrame(sers_rows)
    tran_attr = pd.DataFrame(tran_rows)
    meta_attr = pd.DataFrame(meta_rows)
    fidelity = pd.DataFrame(fidelity_rows)
    random_dist = pd.DataFrame(random_rows)

    contrib.to_csv(args.out_dir / "prediction_contribution_oof.csv", index=False)
    routing.to_csv(args.out_dir / "role_routing_oof.csv", index=False)
    landscape.to_csv(args.out_dir / "consensus_conflict_oof.csv", index=False)
    sers_attr.to_csv(args.out_dir / "sers_attribution_feature_oof.csv", index=False)
    tran_attr.to_csv(args.out_dir / "transcript_attribution_feature_oof.csv", index=False)
    meta_attr.to_csv(args.out_dir / "metabolome_attribution_feature_oof.csv", index=False)
    fidelity.to_csv(args.out_dir / "fidelity_occlusion.csv", index=False)
    random_dist.to_csv(args.out_dir / "fidelity_random_distribution.csv", index=False)

    gate_cols = [f"gate_{name}" for name in ROLE_NAMES]
    summarize_distribution(routing.assign(class_label=routing["label"].map({0: "HC", 1: "RA"})), "class_label", gate_cols).to_csv(args.out_dir / "role_routing_by_class_summary.csv", index=False)
    summarize_distribution(routing.assign(correct_label=np.where(routing["correct"], "correct", "incorrect")), "correct_label", gate_cols).to_csv(args.out_dir / "role_routing_by_correct_summary.csv", index=False)

    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(args.out_dir / "sers_attribution_band_by_fold.csv", index=False)
    band_df.groupby("band").agg(lower_cm1=("lower_cm1", "min"), upper_cm1=("upper_cm1", "max"), mean_abs_attribution=("mean_abs_attribution", "mean"), SD=("mean_abs_attribution", "std"), fold_recurrence=("fold", "nunique"), top5_frequency=("top5", "sum")).reset_index().sort_values(["top5_frequency", "mean_abs_attribution"], ascending=False).to_csv(args.out_dir / "sers_attribution_band_summary.csv", index=False)

    def stability(df: pd.DataFrame, out_name: str):
        top_sets = []
        for _, sub in df.groupby("fold"):
            top_sets.append(set(sub.groupby("feature")["abs_attribution"].mean().sort_values(ascending=False).head(20).index))
        rows = []
        for feature, sub in df.groupby("feature"):
            rows.append({"feature": feature, "mean_abs_attribution": sub["abs_attribution"].mean(), "SD": sub["abs_attribution"].std(ddof=1), "median_abs_attribution": sub["abs_attribution"].median(), "top20_recurrence": sum(feature in s for s in top_sets), "fold_occurrence": sub["fold"].nunique(), "mean_signed_attribution": sub["attribution"].mean()})
        pd.DataFrame(rows).sort_values(["top20_recurrence", "mean_abs_attribution"], ascending=False).to_csv(args.out_dir / out_name, index=False)

    stability(tran_attr, "transcript_feature_stability.csv")
    stability(meta_attr, "metabolome_feature_stability.csv")

    cases = []
    ra_correct = contrib[(contrib["true_label"] == 1) & (contrib["correct"])]
    cases.append({"case_type": "representative_RA", "subject_id": int(ra_correct.sort_values("predicted_probability", ascending=False).iloc[0]["subject_id"])})
    errors = landscape[~landscape["correct"]]
    if len(errors):
        hard = errors.sort_values("pairwise_conflict_mean", ascending=False).iloc[0]
        cases.append({"case_type": "highest_conflict_error", "subject_id": int(hard["subject_id"])})
    else:
        hard = contrib.assign(abs_margin=contrib["total_margin"].abs()).sort_values("abs_margin").iloc[0]
        cases.append({"case_type": "lowest_margin_case", "subject_id": int(hard["subject_id"])})
    pd.DataFrame(cases).to_csv(args.out_dir / "case_selection.csv", index=False)

    report = {"max_reconstruction_error": float(contrib["reconstruction_error"].max()), "n_oof_subjects": int(contrib["subject_id"].nunique()), "n_errors": int((~routing["correct"]).sum())}
    (args.out_dir / "explainability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
