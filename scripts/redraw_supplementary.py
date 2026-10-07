"""Redraw S1/S2 from the supplied aggregate results; no model fitting.

Usage: python redraw_supplementary.py --data-dir data --output-dir .
All plotted means and uncertainty values are copied from the CSV inputs.
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DCE1E5", linewidth=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.14)
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=.14)
    plt.close(fig)


def draw(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": .8, "savefig.facecolor": "white"})
    records = {}
    controls = pd.read_csv(data_dir / "SuppFig1_shortcut.csv")
    perturb = pd.read_csv(data_dir / "SuppFig1_perturbation.csv")
    perturb["norm"] = perturb["norm"].str.upper()
    order = ["baseline", "noise_0.05sd", "shift_+2_index", "shift_-2_index"]
    assert not perturb.duplicated(["norm", "condition"]).any()
    assert set(perturb["norm"]) == {"RAW", "SNV", "L2"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.35))
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.84, wspace=.30)
    a, b = axes
    bars = a.bar(np.arange(3), controls.auc, yerr=controls.sd,
                 color="#287FB0", capsize=4, width=.64,
                 error_kw={"elinewidth": 1, "capthick": 1})
    a.set_xticks(np.arange(3), ["Tri-modal SNV\nreference", "Per-sample\npermutation",
                              "Per-sample\ncircular shift"])
    a.set_ylim(.30, 1.035)
    a.set_ylabel("Five-fold ROC-AUC")
    a.set_title("Reference and SERS-only controls", pad=18, fontsize=11)
    assert np.allclose([p.get_height() for p in bars], controls.auc, rtol=0, atol=0)
    records["S1A"] = [dict(label=l, auc=float(v), sd=float(s)) for l,v,s in zip(
        ["Tri-modal SNV reference", "Per-sample permutation", "Per-sample circular shift"],
        controls.auc, controls.sd)]
    records["S1B"] = {}
    # RAW is uppercase in the source CSV. Normalize before filtering and require all four rows.
    for norm, label, color, marker, offset in [
            ("RAW", "Raw", "#626B75", "s", -.065),
            ("SNV", "SNV", "#3B82F6", "o", 0),
            ("L2", "L2", "#E69B16", "^", .065)]:
        rows = perturb.loc[perturb.norm == norm].set_index("condition").loc[order]
        assert len(rows) == 4 and rows[["auc", "sd"]].notna().all().all()
        x = np.arange(4) + offset
        lines = b.errorbar(x, rows.auc, yerr=rows.sd, marker=marker, color=color,
                          label=label, linewidth=1.45, markersize=5, capsize=3,
                          elinewidth=.85, capthick=.85)
        assert np.array_equal(np.asarray(lines.lines[0].get_ydata()), rows.auc.to_numpy())
        records["S1B"][norm] = rows[["auc", "sd"]].to_dict("index")
    b.set_xticks(np.arange(4), ["Baseline", "0.05 SD\nnoise", "+2 index\nshift", "-2 index\nshift"])
    b.set_ylim(.89, 1.02)
    b.set_xlim(-.25, 3.25)
    b.set_ylabel("Five-fold ROC-AUC")
    b.set_title("Perturbation sensitivity", pad=18, fontsize=11)
    b.legend(loc="lower left", frameon=False, fontsize=9)
    for letter, ax in zip("AB", axes):
        style(ax)
        ax.text(-.13, 1.09, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", ha="left", va="bottom")
    save(fig, output_dir / "SuppFig1_sers_sanity_v011")

    subsets = pd.read_csv(data_dir / "SuppFig2_subset_summary.csv")
    comparisons = pd.read_csv(data_dir / "SuppFig2_incremental_bootstrap.csv")
    order_a = ["all3", "tran_meta", "sers_meta", "sers_tran", "meta", "tran", "sers"]
    labels_a = ["All three modalities", "Transcriptome + metabolome", "SERS + metabolome",
                "SERS + transcriptome", "Metabolome", "Transcriptome", "SERS"]
    order_b = ["add_sers_to_tran_meta", "add_transcript_to_sers_meta", "add_meta_to_sers_tran",
               "add_meta_to_transcript", "add_sers_to_transcript", "add_meta_to_sers",
               "add_transcript_to_sers"]
    labels_b = ["All three vs transcriptome + metabolome", "All three vs SERS + metabolome",
                "All three vs SERS + transcriptome", "Transcriptome + metabolome vs transcriptome",
                "SERS + transcriptome vs transcriptome", "SERS + metabolome vs SERS",
                "SERS + transcriptome vs SERS"]
    sa = subsets.loc[subsets["mode"] == "snv"].set_index("subset").loc[order_a]
    cb = comparisons.loc[comparisons["mode"] == "snv"].set_index("comparison").loc[order_b]
    fig = plt.figure(figsize=(13.8, 5.3))
    a = fig.add_axes([.205, .15, .23, .69])
    b = fig.add_axes([.785, .15, .20, .69])
    y = np.arange(7)
    bars = a.barh(y, sa.auc_mean, xerr=sa.auc_std, color="#287FB0", height=.66,
                  capsize=3, error_kw={"elinewidth": 1})
    a.set_yticks(y, labels_a)
    a.invert_yaxis()
    a.set_xlim(.40, 1.05)
    a.set_xticks([.4,.6,.8,1.0])
    a.set_xlabel("Five-fold ROC-AUC")
    a.set_title("Modality combinations", pad=20, fontsize=11)
    delta = cb.delta_auc_meanfold_a_minus_b.to_numpy()
    err = np.vstack([delta-cb.ci_low.to_numpy(), cb.ci_high.to_numpy()-delta])
    assert np.all(err >= 0)
    points = b.errorbar(delta, y, xerr=err, fmt="o", color="#287FB0", capsize=3,
                       markersize=5, linewidth=1.1)
    b.set_yticks(y, labels_b)
    b.invert_yaxis()
    b.axvline(0, color="#747D85", linewidth=.9)
    b.set_xlim(-.06,.102)
    b.set_xticks([-.05,0,.05,.10])
    b.set_xlabel("Δ ROC-AUC")
    b.set_title("Incremental comparisons", pad=20, fontsize=11)
    for ax in (a,b):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#E5E9EC", linewidth=.6)
    # Figure coordinates keep the panel letters outside the titles and long row labels.
    fig.text(.022, .90, "A", fontsize=13, fontweight="bold")
    fig.text(.475, .90, "B", fontsize=13, fontweight="bold")
    assert np.array_equal(np.asarray([bar.get_width() for bar in bars]), sa.auc_mean.to_numpy())
    assert np.array_equal(np.asarray(points.lines[0].get_xdata()), delta)
    records["S2A"] = sa[["auc_mean", "auc_std"]].to_dict("index")
    records["S2B"] = cb[["delta_auc_meanfold_a_minus_b", "ci_low", "ci_high"]].to_dict("index")
    save(fig, output_dir / "SuppFig2_incremental_modalities_v011")
    (output_dir / "plot_values_verified.json").write_text(json.dumps(records, indent=2) + "\n")
    print("PASS: all plotted means, standard deviations and confidence intervals use the supplied CSV values.")
    print("PASS: RAW, SNV and L2 each contain all four perturbation conditions.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root)
    args = parser.parse_args()
    draw(args.data_dir, args.output_dir)
