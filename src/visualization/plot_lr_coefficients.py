"""
plot_lr_coefficients.py
-----------------------
Extracts logistic regression coefficients from pca_v2_tuned_regime_unaware
for P1_v2, P2_v2, and P3_v2 at label_1m, and generates a bar chart showing
the top positive and negative weights.

Output: data/processed/plots/readme_figures/lr_coefficients.png

Usage:
    python src/visualization/plot_lr_coefficients.py
    python src/visualization/plot_lr_coefficients.py --target label_1y --top-n 20
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

EXPERIMENT = "pca_v2_tuned_regime_unaware"
EXP_DIR    = ROOT / "data" / "processed" / "experiments" / EXPERIMENT
OUT_DIR    = ROOT / "data" / "processed" / "plots" / "readme_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_LABELS = {
    "P1_v2_text_pca":    "P1 — Text + PCA Embeddings",
    "P2_v2_structured":  "P2 — Structured Features",
    "P3_v2_combined_pca":"P3 — Combined (Text + Structured)",
}
VARIANT_COLOURS = {
    "P1_v2_text_pca":    "#4C72B0",
    "P2_v2_structured":  "#DD8452",
    "P3_v2_combined_pca":"#55A868",
}

# Human-readable feature name overrides
RENAME = {
    "sentiment_compound":          "Sentiment (compound)",
    "sentiment_pos":               "Sentiment (positive)",
    "sentiment_neg":               "Sentiment (negative)",
    "uncertainty_density":         "Uncertainty language density",
    "uncertainty_in_risk":         "Uncertainty in risk section",
    "growth_keyword_density":      "Growth keyword density",
    "profit_keyword_density":      "Profitability keyword density",
    "loss_keyword_density":        "Loss/risk keyword density",
    "total_text_length":           "Total filing length",
    "risk_section_length":         "Risk section length",
    "risk_to_total_ratio":         "Risk section / total length",
    "n_sections_found":            "Number of sections found",
    "underwriter_tier":            "Underwriter tier",
    "ipos_prior_30d":              "IPO volume (prior 30d)",
    "ipos_prior_90d":              "IPO volume (prior 90d)",
    "vix_at_ipo":                  "VIX at IPO date",
    "sp500_30d_return":            "S&P 500 trailing 30d return",
    "sp500_90d_return":            "S&P 500 trailing 90d return",
    "sector_etf_30d_return":       "Sector ETF trailing 30d return",
    "revenue_current":             "Revenue (current)",
    "gross_profit":                "Gross profit",
    "net_income":                  "Net income",
    "ebitda":                      "EBITDA",
    "revenue_growth_pct":          "Revenue growth %",
    "gross_margin_pct":            "Gross margin %",
    "insider_proceeds_pct":        "Insider proceeds %",
    "proceeds_debt_pct":           "Proceeds → debt repayment",
    "proceeds_growth_pct":         "Proceeds → growth/R&D",
    "proceeds_general_pct":        "Proceeds → general corporate",
    "proceeds_secondary_pct":      "Proceeds → secondary sale",
    "has_debt_repayment_flag":     "Has debt repayment flag",
    "has_growth_flag":             "Has growth/R&D flag",
    "vix_at_ipo_year_z":           "VIX (year z-score)",
    "vix_at_ipo_year_pctile":      "VIX (year percentile)",
    "vix_at_ipo_roll360_z":        "VIX (360d z-score)",
    "vix_at_ipo_roll360_pctile":   "VIX (360d percentile)",
    "sp500_30d_return_roll360_z":  "S&P 30d return (360d z-score)",
    "sp500_90d_return_roll360_z":  "S&P 90d return (360d z-score)",
    "revenue_growth_pct_roll360_z":"Revenue growth (360d z-score)",
    "gross_margin_pct_roll360_z":  "Gross margin (360d z-score)",
    "gross_margin_pct_roll360_pctile": "Gross margin (360d percentile)",
}


def rename(col: str) -> str:
    if col in RENAME:
        return RENAME[col]
    # PCA components: keep short
    if col.startswith("emb_pc"):
        n = col.replace("emb_pc", "")
        return f"Embedding PC {n}"
    # rolling suffixes → readable
    col = col.replace("_roll360_z", " (360d z)")
    col = col.replace("_roll360_pctile", " (360d pctile)")
    col = col.replace("_year_z", " (yr z)")
    col = col.replace("_year_pctile", " (yr pctile)")
    col = col.replace("_", " ").strip()
    return col


def extract_coefficients(model_path: Path, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    model = joblib.load(model_path)
    coef = model.named_steps["clf"].coef_[0]  # shape (n_features,)
    # Standardise by scaler std so coefficients reflect true feature importance
    scaler = model.named_steps.get("scaler")
    if scaler is not None and hasattr(scaler, "scale_"):
        # Coefficient in original (imputed) space = coef / std
        # But we want the scaled coefs directly — they already encode importance
        # relative to std. Just use raw coef from the scaled space.
        pass
    return coef, [rename(c) for c in feature_cols]


def plot_coefficients(target: str, top_n: int) -> None:
    # Load feature manifest for column names
    manifest_path = EXP_DIR / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    variants = list(VARIANT_LABELS.keys())
    n_variants = len(variants)

    fig, axes = plt.subplots(1, n_variants, figsize=(6.5 * n_variants, max(8, top_n * 0.38 + 2)))
    fig.suptitle(
        f"Logistic Regression Coefficients — {EXPERIMENT}\n"
        f"Target: {target}  |  Top {top_n} positive + negative weights",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, variant in zip(axes, variants):
        feature_cols = manifest["variants"][variant]["feature_columns"]
        model_path = EXP_DIR / "models" / f"{target}_{variant}_logistic_regression.pkl"

        if not model_path.exists():
            ax.set_title(f"{VARIANT_LABELS[variant]}\n(model not found)")
            ax.axis("off")
            continue

        coef, feat_names = extract_coefficients(model_path, feature_cols)

        # Sort and pick top N positive + top N negative
        idx_sorted = np.argsort(coef)
        neg_idx = idx_sorted[:top_n]
        pos_idx = idx_sorted[-top_n:][::-1]
        show_idx = list(pos_idx) + list(neg_idx)

        labels = [feat_names[i] for i in show_idx]
        values = [coef[i]       for i in show_idx]
        colours = [VARIANT_COLOURS[variant] if v > 0 else "#C44E52" for v in values]

        y_pos = np.arange(len(values))
        ax.barh(y_pos, values, color=colours, edgecolor="white", linewidth=0.4, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(VARIANT_LABELS[variant], fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Coefficient (standardised feature space)", fontsize=8)
        ax.invert_yaxis()

        # Divider line between positive and negative halves
        ax.axhline(top_n - 0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        # Add value labels on bars
        for y, val in zip(y_pos, values):
            ax.text(val + (0.002 if val >= 0 else -0.002),
                    y, f"{val:+.3f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=6.5, color="#333")

        # Legend patches
        pos_patch = mpatches.Patch(color=VARIANT_COLOURS[variant], label="Positive (↑ return prob)")
        neg_patch = mpatches.Patch(color="#C44E52", label="Negative (↓ return prob)")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=7.5, loc="lower right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    out = OUT_DIR / f"lr_coefficients_{target}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="label_1m",
                        choices=["label_1w", "label_1m", "label_6m", "label_1y"])
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of top positive + negative coefficients to show per variant")
    args = parser.parse_args()
    plot_coefficients(args.target, args.top_n)
