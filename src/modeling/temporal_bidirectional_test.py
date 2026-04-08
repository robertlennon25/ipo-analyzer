"""
temporal_bidirectional_test.py
------------------------------
Bidirectional temporal generalization test for enhanced experiment variants.

Splits the IPO universe chronologically into two equal halves by IPO date:
  Split A  —  earlier half  (first 50% sorted by ipo_date)
  Split B  —  later  half  (last  50% sorted by ipo_date)

Runs both directions:
  A → B : train on A, evaluate on B
  B → A : train on B, evaluate on A

For each direction × variant × model, reports:
  train_auc, test_auc, auc_drop, test_bal_acc, pred_pos_pct

Feature setup mirrors enhanced_v2_no_ipoyear:
  - Imports load_all_feature_sets / build_variants / EXCLUDE_COLS / _build_model_list
    directly from train_experiment.py so the exclusion list is always in sync.
  - Variants: E1_text_enhanced, E2_structured_enhanced, E3_combined_enhanced

Usage:
  python src/modeling/temporal_bidirectional_test.py
  python src/modeling/temporal_bidirectional_test.py --target label_6m
  python src/modeling/temporal_bidirectional_test.py --split 0.5 --target label_1m

Outputs (nothing existing is overwritten):
  results/temporal_bidirectional/results_{target}.csv   long-form table
  data/processed/experiments/temporal-bidirectional/plots/auc_comparison_{target}.png
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).parent))   # so we can import train_experiment

from config.settings import PROCESSED_DIR  # noqa: E402
from train_experiment import (               # noqa: E402
    load_all_feature_sets,
    build_variants,
    EXCLUDE_COLS,
    _build_model_list,
    HYPERPARAMS,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "temporal_bidirectional"
PLOTS_DIR   = PROCESSED_DIR / "experiments" / "temporal-bidirectional" / "plots"


# ── Helpers ────────────────────────────────────────────────────────────────────

def chronological_split(
    returns_df: pd.DataFrame,
    split_frac: float = 0.5,
) -> tuple[set[str], set[str], str, str]:
    """
    Sort tickers by ipo_date, return (split_A_tickers, split_B_tickers, cutoff_date_A, cutoff_date_B).
    Split A = first split_frac of IPOs by date; Split B = remaining.
    Strict: ties at the boundary go to Split A.
    """
    sorted_df = returns_df[["ticker", "ipo_date"]].dropna().sort_values("ipo_date")
    n_total   = len(sorted_df)
    n_a       = int(np.floor(n_total * split_frac))

    split_a = set(sorted_df.iloc[:n_a]["ticker"])
    split_b = set(sorted_df.iloc[n_a:]["ticker"])

    date_a_end   = sorted_df.iloc[n_a - 1]["ipo_date"].strftime("%Y-%m-%d")
    date_b_start = sorted_df.iloc[n_a]["ipo_date"].strftime("%Y-%m-%d")

    return split_a, split_b, date_a_end, date_b_start


def _predict_scores(model, X: np.ndarray) -> np.ndarray:
    """Return probability or decision scores for binary classification."""
    clf = list(model.named_steps.values())[-1]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)   # Ridge


def evaluate_direction(
    direction: str,
    variant_name: str,
    feat_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    target: str,
    train_tickers: set[str],
    test_tickers: set[str],
) -> list[dict]:
    """
    Train all four models for one direction × variant × target.
    Returns list of result dicts (one per model).
    """
    # Merge labels
    merged = feat_df.merge(returns_df[["ticker", target, "ipo_date"]], on="ticker", how="inner")
    merged = merged.dropna(subset=[target])

    train_df = merged[merged["ticker"].isin(train_tickers)].reset_index(drop=True)
    test_df  = merged[merged["ticker"].isin(test_tickers)].reset_index(drop=True)

    feature_cols = [
        c for c in merged.columns
        if c not in ("ticker", target, "ipo_date")
        and merged[c].dtype in (np.float64, np.float32, np.int64, np.int32)
    ]

    min_samples = HYPERPARAMS["shared"]["min_samples_to_train"]
    if len(train_df) < min_samples or len(test_df) < min_samples:
        logger.warning(
            "Skipping %s / %s / %s — insufficient samples (train=%d, test=%d)",
            direction, variant_name, target, len(train_df), len(test_df),
        )
        return []

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values.astype(int)
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values.astype(int)

    pos_rate         = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    results = []
    for model_name, model in _build_model_list(scale_pos_weight=scale_pos_weight):
        model.fit(X_train, y_train)

        train_scores = _predict_scores(model, X_train)
        test_scores  = _predict_scores(model, X_test)
        test_preds   = model.predict(X_test)

        try:
            train_auc = roc_auc_score(y_train, train_scores)
            test_auc  = roc_auc_score(y_test,  test_scores)
        except ValueError:
            train_auc = test_auc = float("nan")

        bal_acc       = balanced_accuracy_score(y_test, test_preds)
        pred_pos_pct  = round(float(test_preds.mean()) * 100, 1)

        results.append({
            "direction":      direction,
            "variant":        variant_name,
            "target":         target,
            "model":          model_name,
            "train_n":        int(len(y_train)),
            "test_n":         int(len(y_test)),
            "train_pos_rate": round(float(y_train.mean()), 3),
            "test_pos_rate":  round(float(y_test.mean()), 3),
            "train_auc":      round(train_auc, 4),
            "test_auc":       round(test_auc, 4),
            "auc_drop":       round(train_auc - test_auc, 4),
            "test_bal_acc":   round(bal_acc, 4),
            "pred_pos_pct":   pred_pos_pct,
        })

        print(
            f"    {model_name:<22}  train={train_auc:.3f}  "
            f"test={test_auc:.3f}  drop={train_auc-test_auc:+.3f}  "
            f"bal_acc={bal_acc:.3f}"
        )

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_auc_chart(df: pd.DataFrame, target: str, plots_dir: Path) -> Path:
    """
    Grouped bar chart: test AUC for each variant × model, split by direction.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    variants  = df["variant"].unique()
    models    = df["model"].unique()
    directions = ["A_to_B", "B_to_A"]
    colors     = {"A_to_B": "#4C72B0", "B_to_A": "#DD8452"}

    n_variants = len(variants)
    fig, axes  = plt.subplots(1, n_variants, figsize=(6 * n_variants, 5), sharey=True)
    if n_variants == 1:
        axes = [axes]

    x      = np.arange(len(models))
    width  = 0.35

    for ax, variant in zip(axes, variants):
        for i, (direction, offset) in enumerate(zip(directions, [-width / 2, width / 2])):
            sub = df[(df["variant"] == variant) & (df["direction"] == direction)]
            aucs = [sub[sub["model"] == m]["test_auc"].values[0]
                    if len(sub[sub["model"] == m]) else 0.0
                    for m in models]
            bars = ax.bar(x + offset, aucs, width, label=direction, color=colors[direction], alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8, label="random (0.5)")
        ax.set_title(variant, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0.35, 0.85)
        ax.set_ylabel("Test ROC-AUC")
        ax.legend(fontsize=8)

    fig.suptitle(f"Bidirectional Temporal Test — {target}", fontsize=13, y=1.02)
    plt.tight_layout()

    plot_path = plots_dir / f"auc_comparison_{target}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return plot_path


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("BIDIRECTIONAL TEMPORAL TEST — SUMMARY")
    print(f"{'='*70}")

    # Best generalising variant (highest mean test AUC across both directions + all models)
    var_auc = df.groupby("variant")["test_auc"].mean().sort_values(ascending=False)
    print(f"\nBest generalising variant (avg test AUC, both directions):")
    for v, auc in var_auc.items():
        print(f"  {v:<35} {auc:.3f}")

    # Does structured degrade more than text?
    text_drop   = df[df["variant"].str.startswith("E1")]["auc_drop"].mean()
    struct_drop = df[df["variant"].str.startswith("E2")]["auc_drop"].mean()
    comb_drop   = df[df["variant"].str.startswith("E3")]["auc_drop"].mean()
    print(f"\nMean AUC drop (train → test) by variant type:")
    print(f"  E1 text:       {text_drop:+.3f}")
    print(f"  E2 structured: {struct_drop:+.3f}")
    print(f"  E3 combined:   {comb_drop:+.3f}")
    if not np.isnan(struct_drop) and not np.isnan(text_drop):
        if struct_drop > text_drop:
            print("  → Structured features degrade MORE than text (expected: regime sensitivity)")
        elif struct_drop < text_drop:
            print("  → Text features degrade MORE than structured")
        else:
            print("  → Degradation roughly symmetric across types")

    # Symmetry: compare A→B vs B→A per variant
    print(f"\nSymmetry check (A→B test AUC vs B→A test AUC):")
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        atob = sub[sub["direction"] == "A_to_B"]["test_auc"].mean()
        btoa = sub[sub["direction"] == "B_to_A"]["test_auc"].mean()
        diff = atob - btoa
        sign = "+" if diff >= 0 else ""
        sym  = "≈ symmetric" if abs(diff) < 0.02 else ("A→B better" if diff > 0 else "B→A better")
        print(f"  {variant:<35}  A→B={atob:.3f}  B→A={btoa:.3f}  Δ={sign}{diff:.3f}  ({sym})")

    # Best single model overall
    best = df.loc[df["test_auc"].idxmax()]
    print(f"\nBest single run:")
    print(f"  {best['direction']}  {best['variant']}  {best['model']}")
    print(f"  train AUC={best['train_auc']:.3f}  test AUC={best['test_auc']:.3f}  "
          f"drop={best['auc_drop']:+.3f}  bal_acc={best['test_bal_acc']:.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(target: str = "label_1m", split_frac: float = 0.5) -> pd.DataFrame:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    returns_df = pd.read_csv(PROCESSED_DIR / "returns.csv", parse_dates=["ipo_date"])
    feature_sets, _ = load_all_feature_sets()
    variants = build_variants(feature_sets, which="enhanced")

    split_a, split_b, date_a_end, date_b_start = chronological_split(returns_df, split_frac)

    print(f"\nTemporal split  ({split_frac:.0%} / {1-split_frac:.0%})")
    print(f"  Split A: {len(split_a)} IPOs  ≤ {date_a_end}")
    print(f"  Split B: {len(split_b)} IPOs  ≥ {date_b_start}")
    print(f"  Target : {target}")
    print(f"  EXCLUDE_COLS: {EXCLUDE_COLS}\n")

    directions = [
        ("A_to_B", split_a, split_b),
        ("B_to_A", split_b, split_a),
    ]

    all_rows = []
    for direction_label, train_tickers, test_tickers in directions:
        print(f"\n{'─'*60}")
        print(f"Direction: {direction_label}  "
              f"(train n≈{len(train_tickers)}, test n≈{len(test_tickers)})")
        print(f"{'─'*60}")

        for variant_name, feat_df in variants.items():
            print(f"\n  Variant: {variant_name}")
            rows = evaluate_direction(
                direction_label, variant_name, feat_df,
                returns_df, target, train_tickers, test_tickers,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"results_{target}_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved: {results_path}")

    # Save plot
    plot_path = save_auc_chart(df, target, PLOTS_DIR)
    print(f"Plot saved:    {plot_path}")

    print_summary(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bidirectional temporal generalization test")
    parser.add_argument("--target", default="label_1m",
                        help="Return target to evaluate (default: label_1m).")
    parser.add_argument("--split", type=float, default=0.5,
                        help="Fraction of IPOs in Split A (default: 0.5).")
    args = parser.parse_args()

    run(target=args.target, split_frac=args.split)
