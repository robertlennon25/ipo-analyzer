"""
tune_hyperparams.py
-------------------
Randomized hyperparameter search for all model types across experiment variants.
Uses RandomizedSearchCV (5-fold, ROC-AUC) to find better params for tree models
(and optionally linear models). Results are saved so train_experiment.py can
consume them via --hyperparams.

Search spaces:
  random_forest  : max_depth, n_estimators, min_samples_leaf, max_features (40 trials)
  xgboost        : max_depth, n_estimators, learning_rate, subsample, colsample_bytree (40 trials)
  logistic_regression : C (8 values, exhaustive)
  ridge          : alpha (8 values, exhaustive)

Outputs:
  data/processed/experiments/{name}/tuned_params.json
    Structure: { variant_name: { model_name: { param: value, ... }, ... }, ... }
    Consumed by: train_experiment.py --hyperparams <path>

Usage:
  # Tune PCA variants for label_1m
  python src/modeling/tune_hyperparams.py --experiment pca_v1 --variants pca --target label_1m

  # Tune enhanced variants for all targets (slow)
  python src/modeling/tune_hyperparams.py --experiment enhanced_v2_no_ipoyear --variants enhanced

  # Then train with tuned params
  python src/modeling/train_experiment.py \\
      --experiment pca_v1_tuned \\
      --variants pca \\
      --target label_1m \\
      --hyperparams data/processed/experiments/pca_v1/tuned_params.json \\
      --notes "pca_v1 with tuned hyperparams"
"""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PROCESSED_DIR, RANDOM_STATE, CV_FOLDS  # noqa: E402
from train_experiment import (  # noqa: E402
    load_all_feature_sets,
    build_variants,
    HYPERPARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TARGETS    = ["label_1w", "label_1m", "label_6m", "label_1y"]
DEFAULT_EXPERIMENT = "pca_v1"

# ---------------------------------------------------------------------------
# Search spaces — Pipeline params are prefixed with "clf__"
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict] = {
    "random_forest": {
        "clf__max_depth":        [3, 4, 5, 6, 8, None],
        "clf__n_estimators":     [100, 200, 300, 500],
        "clf__min_samples_leaf": [2, 5, 10, 20],
        "clf__max_features":     ["sqrt", "log2", 0.3, 0.5],
    },
    "xgboost": {
        "clf__max_depth":         [2, 3, 4, 5, 6],
        "clf__n_estimators":      [50, 100, 150, 200, 300],
        "clf__learning_rate":     [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample":         [0.6, 0.7, 0.8, 1.0],
        "clf__colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
        "clf__min_child_weight":  [1, 3, 5, 10],
    },
    "logistic_regression": {
        "clf__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    },
    "ridge": {
        "clf__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    },
}

# Number of random draws per model type. Linear models have small grids so fewer needed.
N_ITER: dict[str, int] = {
    "random_forest":      40,
    "xgboost":            40,
    "logistic_regression": 8,
    "ridge":               8,
}


# ---------------------------------------------------------------------------
# Build a fresh base pipeline for searching (no preset hyperparams)
# ---------------------------------------------------------------------------

def _base_pipeline(model_name: str, scale_pos_weight: float = 1.0) -> Pipeline:
    """Return a Pipeline with default-constructed estimator (search will override params)."""
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                 class_weight="balanced")
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", clf)])
    if model_name == "ridge":
        clf = RidgeClassifier()
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("clf", clf)])
    if model_name == "random_forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                     class_weight="balanced")
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("clf", clf)])
    if model_name == "xgboost":
        clf = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                                 verbosity=0, scale_pos_weight=scale_pos_weight)
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("clf", clf)])
    raise ValueError(f"Unknown model: {model_name}")


def _default_cv_score(model_name: str, X: np.ndarray, y: np.ndarray,
                      scale_pos_weight: float) -> float:
    """Score the pipeline with current HYPERPARAMS defaults (for before/after comparison)."""
    from train_experiment import _build_model_list
    from sklearn.model_selection import cross_val_score
    models = dict(_build_model_list(scale_pos_weight=scale_pos_weight))
    pipe = models[model_name]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Tuning for one variant × target
# ---------------------------------------------------------------------------

def tune_variant(
    variant_name: str,
    feat_df: pd.DataFrame,
    target: str,
    returns_df: pd.DataFrame,
) -> dict:
    """
    Run RandomizedSearchCV for all model types on one variant × target.

    Returns:
      {
        model_name: {
          "best_params": { param: value, ... },    # clf__ prefix stripped
          "cv_auc_tuned":   float,
          "cv_auc_default": float,
          "delta":          float,
          "n_trials":       int,
        },
        ...
      }
    """
    merged = feat_df.merge(returns_df[["ticker", target]], on="ticker", how="inner")
    merged = merged.dropna(subset=[target])

    min_samples = HYPERPARAMS["shared"]["min_samples_to_train"]
    if len(merged) < min_samples:
        logger.warning("Skipping %s / %s — insufficient samples (%d)", variant_name, target, len(merged))
        return {}

    y = merged[target].values.astype(int)
    feature_cols = [
        c for c in merged.columns
        if c not in ("ticker", target)
        and merged[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        and not merged[c].isna().all()
    ]
    X = merged[feature_cols].values

    pos_rate         = y.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n  {variant_name} | {target} | "
          f"n={len(y)} features={len(feature_cols)} pos={pos_rate:.1%}")
    print(f"  {'Model':<22} {'Default AUC':<14} {'Tuned AUC':<14} {'Delta':<8} Best params")
    print(f"  {'-'*90}")

    results: dict = {}

    for model_name, param_grid in PARAM_GRIDS.items():
        n_iter = N_ITER[model_name]
        pipe   = _base_pipeline(model_name, scale_pos_weight)

        search = RandomizedSearchCV(
            pipe, param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=False,   # don't refit — we just want the best params
        )
        search.fit(X, y)

        tuned_auc   = float(search.best_score_)
        default_auc = _default_cv_score(model_name, X, y, scale_pos_weight)
        delta       = tuned_auc - default_auc

        # Strip "clf__" prefix from param names for clean storage
        best_params = {
            k.replace("clf__", ""): v
            for k, v in search.best_params_.items()
        }

        results[model_name] = {
            "best_params":    best_params,
            "cv_auc_tuned":   round(tuned_auc, 4),
            "cv_auc_default": round(default_auc, 4),
            "delta":          round(delta, 4),
            "n_trials":       n_iter,
        }

        sign   = "+" if delta >= 0 else ""
        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
        print(f"  {model_name:<22} {default_auc:.4f}         {tuned_auc:.4f}         "
              f"{sign}{delta:.4f}   {params_str}")

    return results


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def run_tuning(
    targets: list[str],
    which: str,
    experiment_name: str,
    notes: str = "",
) -> None:
    exp_dir = PROCESSED_DIR / "experiments" / experiment_name
    if not exp_dir.exists():
        logger.warning(
            "Experiment dir %s does not exist. Tuning results will be saved there anyway.", exp_dir
        )
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTuning experiment : {experiment_name}")
    print(f"Variants          : {which}")
    print(f"Targets           : {targets}")
    if notes:
        print(f"Notes             : {notes}")

    returns_df   = pd.read_csv(PROCESSED_DIR / "returns.csv")
    feature_sets, _ = load_all_feature_sets()
    variants     = build_variants(feature_sets, which=which)

    print(f"\nVariants loaded: {list(variants.keys())}")

    # Accumulate tuning results per target → variant → model
    params_by_target: dict = {}   # { target: { variant: { model: { param: val } } } }
    scores: dict           = {}   # { target: { variant: { model: { auc stats } } } }

    for target in targets:
        print(f"\n{'#'*60}")
        print(f"TARGET: {target}")
        print(f"{'#'*60}")
        scores[target] = {}
        params_by_target[target] = {}

        for variant_name, feat_df in variants.items():
            result = tune_variant(variant_name, feat_df, target, returns_df)
            if not result:
                continue

            scores[target][variant_name] = result

            params_by_target[target][variant_name] = {
                model_name: info["best_params"]
                for model_name, info in result.items()
            }

    # Save per-target params — train_experiment.py detects this format and uses
    # the matching target's params for each training run.
    out_path = exp_dir / "tuned_params.json"
    out_path.write_text(json.dumps(params_by_target, indent=2))
    print(f"\nTuned params saved: {out_path}  (keyed by target)")

    # Save full comparison report
    report: dict = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "notes": notes,
        "targets": targets,
        "variants": which,
        "scores": scores,
        "tuned_params": params_by_target,
    }
    report_path = exp_dir / "tuning_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Tuning report saved: {report_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("TUNING SUMMARY — mean delta AUC (tuned − default) across targets")
    print(f"{'='*70}")
    print(f"{'Variant':<30} {'Model':<22} Avg delta")
    print("-" * 60)
    for variant_name in variants:
        for model_name in PARAM_GRIDS:
            deltas = []
            for target in targets:
                info = scores.get(target, {}).get(variant_name, {}).get(model_name)
                if info:
                    deltas.append(info["delta"])
            if deltas:
                avg = sum(deltas) / len(deltas)
                sign = "+" if avg >= 0 else ""
                print(f"  {variant_name:<28} {model_name:<22} {sign}{avg:.4f}")

    print(f"\nTo train with tuned params:")
    print(f"  python src/modeling/train_experiment.py \\")
    print(f"      --experiment {experiment_name}_tuned \\")
    print(f"      --variants {which} \\")
    print(f"      --hyperparams {out_path} \\")
    print(f"      --notes \"tuned hyperparams from {experiment_name}\"")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for experiment variants")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help="Experiment name — tuned_params.json saved into its directory.")
    parser.add_argument("--variants", default="pca",
                        choices=["baseline", "enhanced", "pca", "pca_v2", "all"],
                        help="Variant family to tune (default: pca).")
    parser.add_argument("--target", default=None,
                        help="Single target (e.g. label_1m). Omit for all targets.")
    parser.add_argument("--notes", default="",
                        help="Optional description attached to the tuning report.")
    args = parser.parse_args()

    targets = [args.target] if args.target else DEFAULT_TARGETS
    run_tuning(
        targets=targets,
        which=args.variants,
        experiment_name=args.experiment,
        notes=args.notes,
    )
