"""
generate_readme_figures.py
--------------------------
Generates the four figures referenced in README.md:

  1. auc_progression.png       — Best CV AUC per experiment version (M→E→PCA v1→PCA v2)
                                  by feature group (text / structured / combined)
  2. overfit_scatter.png        — Train AUC vs temporal test AUC; size = AUC drop
  3. auc_by_horizon.png         — Grouped bar chart: P1_v2 / P2_v2 / P3_v2 across all 4 horizons
  4. temporal_asymmetry.png     — A→B vs B→A test AUC scatter for pca_v2 variants

Output: data/processed/plots/readme_figures/
"""

import sys
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

OUTPUT_DIR = ROOT / "data" / "processed" / "plots" / "readme_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
PALETTE = {
    "text":       "#4C72B0",   # blue
    "structured": "#DD8452",   # orange
    "combined":   "#55A868",   # green
}
MODEL_MARKERS = {
    "logistic_regression": "o",
    "ridge":               "s",
    "random_forest":       "^",
    "xgboost":             "D",
}
MODEL_COLOURS = {
    "logistic_regression": "#4C72B0",
    "ridge":               "#55A868",
    "random_forest":       "#DD8452",
    "xgboost":             "#C44E52",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

# ---------------------------------------------------------------------------
# Helper: load best CV AUC per variant-group from a run JSON
# ---------------------------------------------------------------------------
def _best_per_group(run_path: Path, group_map: dict) -> dict[str, float]:
    """
    group_map: { variant_name → group }  e.g. {"M1_text": "text", ...}
    Returns {group → best_auc_across_all_models_and_targets_listed}.
    We focus on label_1m for the progression chart.
    """
    with open(run_path) as f:
        r = json.load(f)

    results = r.get("results", {})
    best: dict[str, float] = {}
    target = "label_1m"
    tdata = results.get(target, {})
    for variant_name, group in group_map.items():
        vdata = tdata.get("variants", {}).get(variant_name, {})
        for model, mdata in vdata.items():
            auc = mdata.get("roc_auc_mean", 0)
            if auc > best.get(group, 0):
                best[group] = round(auc, 4)
    return best


# ---------------------------------------------------------------------------
# Figure 1 — AUC Progression
# ---------------------------------------------------------------------------
def fig_auc_progression():
    versions = ["Baseline\n(M-series)", "Enhanced\n(E-series)", "PCA v1\n(tuned)", "PCA v2\n(tuned)"]

    run_paths = {
        "baseline":  ROOT / "data/processed/run_results/run_20260407_225709.json",
        "enhanced":  ROOT / "data/processed/experiments/enhanced_v2_no_ipoyear/run_results/run_20260407_234544.json",
        "pca_v1t":   ROOT / "data/processed/experiments/pca_v1_tuned/run_results/run_20260408_152848.json",
        "pca_v2t":   ROOT / "data/processed/experiments/pca_v2_tuned_regime_unaware/run_results/run_20260408_152536.json",
    }
    group_maps = {
        "baseline": {"M1_text": "text", "M2_multiples": "structured", "M3_combined": "combined"},
        "enhanced": {"E1_text_enhanced": "text", "E2_structured_enhanced": "structured", "E3_combined_enhanced": "combined"},
        "pca_v1t":  {"P1_text_pca": "text", "P2_structured": "structured", "P3_combined_pca": "combined"},
        "pca_v2t":  {"P1_v2_text_pca": "text", "P2_v2_structured": "structured", "P3_v2_combined_pca": "combined"},
    }

    data: dict[str, list[float]] = {"text": [], "structured": [], "combined": []}
    for key, path in run_paths.items():
        best = _best_per_group(path, group_maps[key])
        for g in data:
            data[g].append(best.get(g, np.nan))

    x = np.arange(len(versions))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for group, aucs in data.items():
        ax.plot(x, aucs, "o-", color=PALETTE[group], linewidth=2.2,
                markersize=7, label=group.capitalize(), zorder=3)
        for xi, auc in zip(x, aucs):
            if not np.isnan(auc):
                ax.annotate(f"{auc:.3f}", (xi, auc),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8.5, color=PALETTE[group])

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=10)
    ax.set_ylabel("Best CV ROC-AUC (label_1m)", fontsize=10)
    ax.set_title("AUC Progression Across Experiment Versions", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.48, 0.72)

    # Annotation arrows / notes
    ax.annotate("ipo_year\nleakage removed", xy=(1, data["structured"][1]),
                xytext=(1.15, 0.595), fontsize=7.5, color="#888",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    ax.annotate("PCA compression\n(384→30 dims)", xy=(2, data["text"][2]),
                xytext=(2.1, 0.628), fontsize=7.5, color="#888",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))

    fig.tight_layout()
    out = OUTPUT_DIR / "auc_progression.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Overfitting Scatter (train AUC vs test AUC)
# ---------------------------------------------------------------------------
def fig_overfit_scatter():
    # Use pca_v2 bidirectional results for label_1m (most complete)
    df = pd.read_csv(ROOT / "results/temporal_bidirectional/results_label_1m_20260408_203052.csv")

    # Average both directions per variant × model
    agg = df.groupby(["variant", "model"]).agg(
        train_auc=("train_auc", "mean"),
        test_auc=("test_auc", "mean"),
    ).reset_index()

    # Map variant → group
    group_map = {
        "P1_v2_text_pca":    "text",
        "P2_v2_structured":  "structured",
        "P3_v2_combined_pca":"combined",
    }

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for _, row in agg.iterrows():
        group = group_map.get(row["variant"], "combined")
        marker = MODEL_MARKERS.get(row["model"], "o")
        ax.scatter(row["train_auc"], row["test_auc"],
                   color=PALETTE[group], marker=marker, s=90,
                   edgecolors="white", linewidth=0.5, zorder=3)

    # Diagonal reference line (no overfit)
    lims = [0.45, 1.02]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, alpha=0.6, label="No overfit (train=test)")

    # Shade overfit zone
    ax.fill_between(lims, lims, [lims[0], lims[0]], alpha=0.04, color="red")
    ax.text(0.87, 0.505, "Overfitting zone", fontsize=7.5, color="#bbb", rotation=0)

    # Legend: groups
    group_patches = [mpatches.Patch(color=c, label=g.capitalize())
                     for g, c in PALETTE.items()]
    # Legend: models
    model_handles = [
        plt.Line2D([0], [0], marker=m, color="gray", linestyle="None",
                   markersize=7, label=label.replace("_", " ").title())
        for label, m in MODEL_MARKERS.items()
    ]
    leg1 = ax.legend(handles=group_patches, loc="upper left", fontsize=8.5, title="Feature group")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="lower right", fontsize=8.5, title="Model")

    ax.set_xlabel("Train AUC (avg across both temporal directions)", fontsize=10)
    ax.set_ylabel("Test AUC (avg across both temporal directions)", fontsize=10)
    ax.set_title("Overfitting: Train vs Temporal Test AUC\npca_v2 variants, label_1m",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0.75, 1.02)
    ax.set_ylim(0.46, 0.66)

    # Annotate the best generalizers
    for _, row in agg.iterrows():
        if row["test_auc"] > 0.60:
            label = f"{row['model'].split('_')[0]}\n({row['variant'].split('_')[0]})"
            ax.annotate(label, (row["train_auc"], row["test_auc"]),
                        xytext=(-36, 8), textcoords="offset points",
                        fontsize=7, color="#444",
                        arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.7))

    fig.tight_layout()
    out = OUTPUT_DIR / "overfit_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — AUC by Horizon
# ---------------------------------------------------------------------------
def fig_auc_by_horizon():
    run_path = ROOT / "data/processed/experiments/pca_v2_tuned_regime_unaware/run_results/run_20260408_152536.json"
    with open(run_path) as f:
        r = json.load(f)

    targets = ["label_1w", "label_1m", "label_6m", "label_1y"]
    horizon_labels = ["1 Week", "1 Month", "6 Months", "1 Year"]
    variants = {
        "P1_v2_text_pca":     "text",
        "P2_v2_structured":   "structured",
        "P3_v2_combined_pca": "combined",
    }

    # Best AUC per variant × target
    best: dict[str, list[float]] = {g: [] for g in ["text", "structured", "combined"]}
    for target in targets:
        tdata = r["results"].get(target, {}).get("variants", {})
        for vname, group in variants.items():
            vdata = tdata.get(vname, {})
            top = max((m.get("roc_auc_mean", 0) for m in vdata.values()), default=np.nan)
            best[group].append(round(top, 4))

    x = np.arange(len(targets))
    width = 0.26
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for i, (group, aucs) in enumerate(best.items()):
        bars = ax.bar(x + offsets[i], aucs, width=width - 0.02,
                      color=PALETTE[group], label=group.capitalize(),
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{auc:.3f}", ha="center", va="bottom", fontsize=7.5,
                    color=PALETTE[group], fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels, fontsize=11)
    ax.set_ylabel("Best CV ROC-AUC", fontsize=10)
    ax.set_title("Signal Strength by Return Horizon\npca_v2_tuned — best model per variant × horizon",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0.47, 0.80)

    fig.tight_layout()
    out = OUTPUT_DIR / "auc_by_horizon.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4 — Temporal Asymmetry (A→B vs B→A)
# ---------------------------------------------------------------------------
def fig_temporal_asymmetry():
    # Use pca_v2 label_1m bidirectional results
    df = pd.read_csv(ROOT / "results/temporal_bidirectional/results_label_1m_20260408_203052.csv")

    a_to_b = df[df["direction"] == "A_to_B"][["variant", "model", "test_auc"]].rename(
        columns={"test_auc": "atob"})
    b_to_a = df[df["direction"] == "B_to_A"][["variant", "model", "test_auc"]].rename(
        columns={"test_auc": "btoa"})
    merged = a_to_b.merge(b_to_a, on=["variant", "model"])

    group_map = {
        "P1_v2_text_pca":    "text",
        "P2_v2_structured":  "structured",
        "P3_v2_combined_pca":"combined",
    }

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for _, row in merged.iterrows():
        group = group_map.get(row["variant"], "combined")
        marker = MODEL_MARKERS.get(row["model"], "o")
        ax.scatter(row["atob"], row["btoa"],
                   color=PALETTE[group], marker=marker, s=90,
                   edgecolors="white", linewidth=0.5, zorder=3)

    # Symmetry line
    lims = [0.48, 0.66]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, alpha=0.6, label="Perfect symmetry")

    # Shade region where A→B > B→A (typical direction)
    xs = np.linspace(lims[0], lims[1], 100)
    ax.fill_between(xs, xs, lims[0], alpha=0.04, color=PALETTE["text"])
    ax.text(0.583, 0.493, "A→B dominant\n(early → late generalises better)",
            fontsize=7.5, color="#999", ha="center")

    # Legend
    group_patches = [mpatches.Patch(color=c, label=g.capitalize())
                     for g, c in PALETTE.items()]
    model_handles = [
        plt.Line2D([0], [0], marker=m, color="gray", linestyle="None",
                   markersize=7, label=label.replace("_", " ").title())
        for label, m in MODEL_MARKERS.items()
    ]
    leg1 = ax.legend(handles=group_patches, loc="upper left", fontsize=8.5, title="Feature group")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="lower right", fontsize=8.5, title="Model")

    ax.set_xlabel("A→B Test AUC  (train pre-2021, test post-2021)", fontsize=9.5)
    ax.set_ylabel("B→A Test AUC  (train post-2021, test pre-2021)", fontsize=9.5)
    ax.set_title("Temporal Asymmetry: A→B vs B→A Generalization\npca_v2 variants, label_1m",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    fig.tight_layout()
    out = OUTPUT_DIR / "temporal_asymmetry.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}\n")
    fig_auc_progression()
    fig_overfit_scatter()
    fig_auc_by_horizon()
    fig_temporal_asymmetry()
    print("\nAll figures generated.")
