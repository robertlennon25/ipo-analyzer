"""
compare_experiments.py
----------------------
Compare model performance across experiment families (baseline vs enhanced).

Loads one or more run JSON files and produces:
  1. Long-form CSV  — one row per experiment × variant × target × model
  2. Pivoted CSV    — AUC/Bal-Acc side-by-side for easy reading
  3. Console summary — best model per target, delta AUC vs baseline

Delta columns (enhanced vs baseline):
  - Matches by (target, model_type, variant_group)
  - variant_group mapping:
      M1_text            → text
      E1_text_enhanced   → text
      M2_multiples       → structured
      E2_structured_enhanced → structured
      M3_combined        → combined
      E3_combined_enhanced   → combined

Usage:
    # Auto-discover latest baseline + latest enhanced_v1 run
    python src/modeling/compare_experiments.py

    # Specify run files explicitly
    python src/modeling/compare_experiments.py \\
        --baseline data/processed/run_results/run_20260407_225709.json \\
        --enhanced data/processed/experiments/enhanced_v1/run_results/run_XXXXXXXX.json

    # Custom experiment name and output tag
    python src/modeling/compare_experiments.py \\
        --experiment enhanced_v1 \\
        --tag my_comparison

Outputs (saved to data/processed/comparisons/{tag}/):
    comparison_long.csv       one row per experiment × variant × target × model
    comparison_pivot.csv      AUC side-by-side, easy to read
    comparison_config.json    which run files were compared
"""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COMPARISONS_DIR = PROCESSED_DIR / "comparisons"

# Maps variant name → comparison group (for delta computation)
VARIANT_GROUP: dict[str, str] = {
    "M1_text":               "text",
    "E1_text_enhanced":      "text",
    "M2_multiples":          "structured",
    "E2_structured_enhanced": "structured",
    "M3_combined":           "combined",
    "E3_combined_enhanced":  "combined",
}


# ---------------------------------------------------------------------------
# Run file discovery
# ---------------------------------------------------------------------------

def _latest_run(directory: Path) -> Path | None:
    """Return the most recent run_*.json in directory, or None."""
    files = sorted(directory.glob("run_*.json"))
    return files[-1] if files else None


def _load_run(path: Path) -> dict:
    with open(path) as f:
        envelope = json.load(f)
    # Handle both plain results dicts and run envelopes
    if "results" not in envelope:
        envelope = {"run_id": "unknown", "experiment": "unknown", "results": envelope}
    return envelope


def discover_runs(
    baseline_path: str | None,
    experiment_name: str,
    enhanced_path: str | None,
) -> tuple[dict, dict, Path, Path]:
    """
    Resolve run files for baseline and enhanced experiments.
    Returns (baseline_envelope, enhanced_envelope, baseline_path, enhanced_path).
    """
    # --- Baseline ---
    if baseline_path:
        bp = Path(baseline_path)
    else:
        bp = _latest_run(PROCESSED_DIR / "run_results")
        if bp is None:
            raise FileNotFoundError(
                "No run files found in data/processed/run_results/. "
                "Run train.py first, or pass --baseline explicitly."
            )
        print(f"Auto-selected baseline run: {bp.name}")

    # --- Enhanced ---
    if enhanced_path:
        ep = Path(enhanced_path)
    else:
        ep = _latest_run(PROCESSED_DIR / "experiments" / experiment_name / "run_results")
        if ep is None:
            raise FileNotFoundError(
                f"No run files found in data/processed/experiments/{experiment_name}/run_results/. "
                "Run train_experiment.py first, or pass --enhanced explicitly."
            )
        print(f"Auto-selected enhanced run: {ep.name}")

    return _load_run(bp), _load_run(ep), bp, ep


# ---------------------------------------------------------------------------
# Flatten run JSON → long-form DataFrame
# ---------------------------------------------------------------------------

def _flatten_run(envelope: dict, family_label: str) -> pd.DataFrame:
    """
    Flatten a run envelope's results into a long-form DataFrame.
    family_label: e.g. "baseline" or "enhanced"
    """
    rows = []
    results = envelope.get("results", {})
    run_id  = envelope.get("run_id", "unknown")
    experiment = envelope.get("experiment", family_label)

    for target, target_data in results.items():
        variants = target_data.get("variants", {})
        for variant, models in variants.items():
            for model_name, m in models.items():
                rows.append({
                    "run_id":            run_id,
                    "experiment":        experiment,
                    "family":            family_label,
                    "variant":           variant,
                    "variant_group":     VARIANT_GROUP.get(variant, "other"),
                    "target":            target,
                    "model":             model_name,
                    "roc_auc_mean":      m.get("roc_auc_mean"),
                    "roc_auc_std":       m.get("roc_auc_std"),
                    "bal_accuracy_mean": m.get("bal_accuracy_mean", m.get("accuracy_mean")),
                    "bal_accuracy_std":  m.get("bal_accuracy_std", m.get("accuracy_std")),
                    "naive_accuracy":    m.get("naive_accuracy"),
                    "pred_positive_pct": m.get("pred_positive_pct"),
                    "n_samples":         m.get("n_samples"),
                    "n_features":        m.get("n_features"),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta_auc_vs_baseline and delta_bal_acc_vs_baseline columns.
    Matches enhanced rows to baseline rows by (target, model, variant_group).
    """
    baseline = long_df[long_df["family"] == "baseline"].copy()
    enhanced = long_df[long_df["family"] == "enhanced"].copy()

    if baseline.empty or enhanced.empty:
        long_df["delta_auc_vs_baseline"]     = np.nan
        long_df["delta_bal_acc_vs_baseline"] = np.nan
        return long_df

    ref = baseline[["target", "model", "variant_group", "roc_auc_mean", "bal_accuracy_mean"]].rename(
        columns={"roc_auc_mean": "ref_auc", "bal_accuracy_mean": "ref_bal_acc"}
    )

    merged = long_df.merge(ref, on=["target", "model", "variant_group"], how="left")
    merged["delta_auc_vs_baseline"]     = merged["roc_auc_mean"]      - merged["ref_auc"]
    merged["delta_bal_acc_vs_baseline"] = merged["bal_accuracy_mean"] - merged["ref_bal_acc"]
    # Baseline rows get delta=0 (vs themselves)
    baseline_mask = merged["family"] == "baseline"
    merged.loc[baseline_mask, "delta_auc_vs_baseline"]     = 0.0
    merged.loc[baseline_mask, "delta_bal_acc_vs_baseline"] = 0.0

    return merged.drop(columns=["ref_auc", "ref_bal_acc"])


# ---------------------------------------------------------------------------
# Pivot table
# ---------------------------------------------------------------------------

def make_pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    """Wide-format table: one row per (target, model, variant_group), columns per family."""
    rows = []
    for (target, model, vgroup), grp in long_df.groupby(["target", "model", "variant_group"]):
        row: dict = {"target": target, "model": model, "variant_group": vgroup}
        for _, r in grp.iterrows():
            prefix = r["family"]
            row[f"{prefix}_variant"]      = r["variant"]
            row[f"{prefix}_auc"]          = r["roc_auc_mean"]
            row[f"{prefix}_auc_std"]      = r["roc_auc_std"]
            row[f"{prefix}_bal_acc"]      = r["bal_accuracy_mean"]
            row[f"{prefix}_n_samples"]    = r["n_samples"]
            row[f"{prefix}_n_features"]   = r["n_features"]
            if not pd.isna(r.get("delta_auc_vs_baseline", np.nan)):
                row["delta_auc"]     = r["delta_auc_vs_baseline"]
                row["delta_bal_acc"] = r["delta_bal_acc_vs_baseline"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(long_df: pd.DataFrame) -> None:
    targets = sorted(long_df["target"].unique())

    for target in targets:
        sub = long_df[long_df["target"] == target]
        print(f"\n{'='*80}")
        print(f"TARGET: {target}")
        print(f"{'='*80}")

        for family in ("baseline", "enhanced"):
            fam_sub = sub[sub["family"] == family]
            if fam_sub.empty:
                continue
            best = fam_sub.loc[fam_sub["roc_auc_mean"].idxmax()]
            print(
                f"  Best {family:8s}: {best['variant']:<28} {best['model']:<22} "
                f"AUC={best['roc_auc_mean']:.3f}±{best['roc_auc_std']:.3f}  "
                f"Bal-Acc={best['bal_accuracy_mean']:.3f}"
            )

        # Delta summary per variant group
        enh = sub[sub["family"] == "enhanced"]
        if not enh.empty and "delta_auc_vs_baseline" in enh.columns:
            print(f"\n  Delta AUC (enhanced − baseline) by variant group:")
            for vgroup in sorted(enh["variant_group"].unique()):
                vg_sub = enh[enh["variant_group"] == vgroup]
                best_delta = vg_sub.loc[vg_sub["delta_auc_vs_baseline"].abs().idxmax()]
                d = best_delta["delta_auc_vs_baseline"]
                sign = "+" if d >= 0 else ""
                print(
                    f"    {vgroup:<12}: best delta = {sign}{d:+.3f} AUC "
                    f"({best_delta['model']}, {best_delta['variant']})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(
    baseline_path: str | None = None,
    enhanced_path: str | None = None,
    experiment_name: str = "enhanced_v1",
    tag: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load baseline and enhanced run files, compute comparison, save outputs.
    Returns {"long": long_df, "pivot": pivot_df}.
    """
    baseline_env, enhanced_env, bp, ep = discover_runs(
        baseline_path, experiment_name, enhanced_path
    )

    baseline_df = _flatten_run(baseline_env, "baseline")
    enhanced_df = _flatten_run(enhanced_env, "enhanced")
    long_df     = pd.concat([baseline_df, enhanced_df], ignore_index=True)
    long_df     = compute_deltas(long_df)
    pivot_df    = make_pivot(long_df)

    # Output directory
    out_tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = COMPARISONS_DIR / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path  = out_dir / "comparison_long.csv"
    pivot_path = out_dir / "comparison_pivot.csv"
    cfg_path   = out_dir / "comparison_config.json"

    long_df.to_csv(long_path,  index=False)
    pivot_df.to_csv(pivot_path, index=False)

    cfg = {
        "generated_at":    datetime.now().isoformat(),
        "baseline_run":    str(bp),
        "enhanced_run":    str(ep),
        "experiment_name": experiment_name,
        "tag":             out_tag,
        "baseline_run_id": baseline_env.get("run_id"),
        "enhanced_run_id": enhanced_env.get("run_id"),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))

    print_console_summary(long_df)

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  {long_path.name}   — {len(long_df)} rows")
    print(f"  {pivot_path.name}  — {len(pivot_df)} rows")
    print(f"  {cfg_path.name}")

    return {"long": long_df, "pivot": pivot_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare baseline vs enhanced IPO model experiments")
    parser.add_argument("--baseline",   default=None,
                        help="Path to baseline run JSON. Defaults to latest in run_results/.")
    parser.add_argument("--enhanced",   default=None,
                        help="Path to enhanced run JSON. Defaults to latest in experiments/{experiment}/run_results/.")
    parser.add_argument("--experiment", default="enhanced_v1",
                        help="Experiment name (used to locate enhanced run files).")
    parser.add_argument("--tag",        default=None,
                        help="Tag for output directory under comparisons/. Defaults to timestamp.")
    args = parser.parse_args()

    run_comparison(
        baseline_path=args.baseline,
        enhanced_path=args.enhanced,
        experiment_name=args.experiment,
        tag=args.tag,
    )
