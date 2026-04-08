"""
train_experiment.py
-------------------
Train model variants for a named experiment and save all outputs under a
versioned directory — no existing files are overwritten.

Experiment variants:
  Baseline (reproduced for apples-to-apples comparison):
    M1_text           handcrafted + embeddings
    M2_multiples      multiples + market context
    M3_combined       M1 + M2

  Enhanced (new features):
    E1_text_enhanced        M1 features + proceeds
    E2_structured_enhanced  M2 features + underwriter + proceeds + regime_normalized
    E3_combined_enhanced    E1 + E2

Usage:
    # Enhanced variants only (default)
    python src/modeling/train_experiment.py

    # Baseline variants only
    python src/modeling/train_experiment.py --variants baseline

    # Both families (useful for fresh apples-to-apples comparison)
    python src/modeling/train_experiment.py --variants all

    # Single target, custom experiment name, with notes
    python src/modeling/train_experiment.py \\
        --target label_1m \\
        --experiment enhanced_v1 \\
        --variants enhanced \\
        --notes "first run with underwriter + proceeds + regime features"

Output layout (example --experiment enhanced_v1):
    data/processed/experiments/enhanced_v1/
        feature_manifest.json                      feature columns per variant + sources
        run_results/run_YYYYMMDD_HHMMSS.json       full run envelope (never overwritten)
        models/{target}_{variant}_{model}.pkl      trained models
"""

import sys
import json
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, RANDOM_STATE, CV_FOLDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TARGETS    = ["label_1w", "label_1m", "label_6m", "label_1y"]
DEFAULT_EXPERIMENT = "enhanced_v1"

# ---------------------------------------------------------------------------
# Columns to exclude from all experiment feature matrices.
# ipo_year is a raw integer that leaks the calendar year directly to the model,
# enabling temporal shortcut learning (model learns "2021 = good" rather than
# any filing or market signal). The regime_normalized features (*_year_z,
# *_year_pctile) are computed from ipo_year but are cross-sectionally
# normalised, so they are retained.
# ---------------------------------------------------------------------------
EXCLUDE_COLS: set[str] = {"ipo_year"}

# ---------------------------------------------------------------------------
# Hyperparameters — single source of truth serialised into every run JSON.
# Keep in sync with train.py if you change values there.
# ---------------------------------------------------------------------------
HYPERPARAMS: dict = {
    "logistic_regression": {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"},
    "ridge":               {"alpha": 1.0, "note": "no class_weight — caution on imbalanced targets"},
    "random_forest":       {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5, "class_weight": "balanced"},
    "xgboost":             {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "eval_metric": "logloss",
                            "note": "scale_pos_weight computed per-target from class ratio"},
    "shared":              {"imputer_strategy": "median", "cv_folds": CV_FOLDS,
                            "random_state": RANDOM_STATE, "min_samples_to_train": 30},
}


def _build_model_list(scale_pos_weight: float = 1.0) -> list[tuple[str, Pipeline]]:
    hp = HYPERPARAMS
    return [
        ("logistic_regression", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(
                max_iter=hp["logistic_regression"]["max_iter"],
                random_state=RANDOM_STATE, C=hp["logistic_regression"]["C"],
                class_weight=hp["logistic_regression"]["class_weight"],
            )),
        ])),
        ("ridge", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RidgeClassifier(alpha=hp["ridge"]["alpha"])),
        ])),
        ("random_forest", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(
                n_estimators=hp["random_forest"]["n_estimators"],
                max_depth=hp["random_forest"]["max_depth"],
                min_samples_leaf=hp["random_forest"]["min_samples_leaf"],
                random_state=RANDOM_STATE, n_jobs=-1,
                class_weight=hp["random_forest"]["class_weight"],
            )),
        ])),
        ("xgboost", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     xgb.XGBClassifier(
                n_estimators=hp["xgboost"]["n_estimators"],
                max_depth=hp["xgboost"]["max_depth"],
                learning_rate=hp["xgboost"]["learning_rate"],
                random_state=RANDOM_STATE, eval_metric=hp["xgboost"]["eval_metric"],
                verbosity=0, scale_pos_weight=scale_pos_weight,
            )),
        ])),
    ]


def _evaluate_model(model, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Cross-validate model and return metrics dict. Fits final model in place."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auc_scores     = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    acc_scores     = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    bal_acc_scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
    oof_preds      = cross_val_predict(model, X, y, cv=cv)
    model.fit(X, y)

    pred_pos = float(oof_preds.mean())
    result = {
        "roc_auc_mean":      float(auc_scores.mean()),
        "roc_auc_std":       float(auc_scores.std()),
        "accuracy_mean":     float(acc_scores.mean()),
        "accuracy_std":      float(acc_scores.std()),
        "bal_accuracy_mean": float(bal_acc_scores.mean()),
        "bal_accuracy_std":  float(bal_acc_scores.std()),
        "pred_positive_pct": round(pred_pos * 100, 1),
        "pred_negative_pct": round((1 - pred_pos) * 100, 1),
        "n_samples":         int(len(y)),
        "n_features":        int(X.shape[1]),
        "positive_rate":     float(y.mean()),
        "naive_accuracy":    float(max(y.mean(), 1 - y.mean())),
    }

    final = list(model.named_steps.values())[-1]
    importance = None
    if hasattr(final, "feature_importances_"):
        importance = final.feature_importances_
    elif hasattr(final, "coef_"):
        coef = final.coef_
        importance = np.abs(coef[0] if coef.ndim > 1 else coef)

    if importance is not None and len(feature_names) == len(importance):
        top_idx = np.argsort(importance)[::-1][:20]
        result["top_features"] = [
            {"feature": feature_names[i], "importance": float(importance[i])}
            for i in top_idx
        ]
    return result


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop object columns (except ticker) and any explicitly excluded columns."""
    drop = [
        c for c in df.columns
        if (df[c].dtype == object and c != "ticker") or c in EXCLUDE_COLS
    ]
    return df.drop(columns=drop)


def _merge(left: pd.DataFrame, right: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Merge two feature DataFrames on 'ticker', avoiding duplicate column names."""
    existing = set(left.columns) - {"ticker"}
    keep = ["ticker"] + [c for c in right.columns if c not in existing and c != "ticker"]
    return left.merge(right[keep], on="ticker", how=how)


def load_all_feature_sets() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Load all available feature CSVs.
    Returns (feature_sets dict, source_paths dict).
    Warns if enhanced feature files are missing.
    """
    fs: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}

    def _load(key: str, path: Path, required: bool = False) -> None:
        if path.exists():
            fs[key] = pd.read_csv(path)
            sources[key] = str(path)
        elif required:
            raise FileNotFoundError(f"Required feature file missing: {path}")
        else:
            logger.warning("Feature file not found (skipping): %s", path)

    # Baseline features
    _load("handcrafted",      PROCESSED_DIR / "handcrafted_features.csv")
    _load("multiples",        PROCESSED_DIR / "multiples_features.csv")
    _load("market",           PROCESSED_DIR / "market_context_features.csv")

    try:
        from src.features.embeddings import load_embeddings, embeddings_to_dataframe
        embeddings, tickers = load_embeddings()
        fs["embeddings"] = embeddings_to_dataframe(embeddings, tickers)
        sources["embeddings"] = str(PROCESSED_DIR.parent / "cache" / "embeddings.npz")
    except FileNotFoundError:
        logger.warning("Embeddings not found — skipping. Run embeddings.py first.")

    # Enhanced features
    _load("underwriter",      PROCESSED_DIR / "underwriter_features.csv")
    _load("proceeds",         PROCESSED_DIR / "proceeds_features.csv")
    _load("regime_normalized", PROCESSED_DIR / "regime_normalized_features.csv")

    missing_enhanced = [k for k in ("underwriter", "proceeds", "regime_normalized") if k not in fs]
    if missing_enhanced:
        print(f"\nWARNING: Missing enhanced features: {missing_enhanced}")
        print("Run the corresponding feature scripts before training enhanced variants.\n")

    return fs, sources


def build_variants(
    feature_sets: dict[str, pd.DataFrame],
    which: str = "enhanced",
) -> dict[str, pd.DataFrame]:
    """
    Assemble feature matrices for requested variant family.
    which: "baseline" | "enhanced" | "all"
    """
    v: dict[str, pd.DataFrame] = {}

    if which in ("baseline", "all"):
        # M1_text
        parts = []
        if "handcrafted" in feature_sets:
            parts.append(_clean(feature_sets["handcrafted"]))
        if "embeddings" in feature_sets:
            parts.append(feature_sets["embeddings"])
        if parts:
            m1 = parts[0]
            for p in parts[1:]:
                m1 = _merge(m1, p, how="inner")
            v["M1_text"] = m1

        # M2_multiples
        parts = []
        if "multiples" in feature_sets:
            parts.append(_clean(feature_sets["multiples"]))
        if "market" in feature_sets:
            parts.append(_clean(feature_sets["market"]))
        if parts:
            m2 = parts[0]
            for p in parts[1:]:
                m2 = _merge(m2, p, how="outer")
            v["M2_multiples"] = m2

        # M3_combined
        if "M1_text" in v and "M2_multiples" in v:
            v["M3_combined"] = _merge(v["M1_text"], v["M2_multiples"], how="inner")

    if which in ("enhanced", "all"):
        # E1_text_enhanced: M1 features + proceeds
        parts = []
        if "handcrafted" in feature_sets:
            parts.append(_clean(feature_sets["handcrafted"]))
        if "embeddings" in feature_sets:
            parts.append(feature_sets["embeddings"])
        if "proceeds" in feature_sets:
            parts.append(_clean(feature_sets["proceeds"]))
        if parts:
            e1 = parts[0]
            for p in parts[1:]:
                e1 = _merge(e1, p, how="left")
            v["E1_text_enhanced"] = e1

        # E2_structured_enhanced: M2 features + underwriter + proceeds + regime_normalized
        parts = []
        if "multiples" in feature_sets:
            parts.append(_clean(feature_sets["multiples"]))
        if "market" in feature_sets:
            parts.append(_clean(feature_sets["market"]))
        if "underwriter" in feature_sets:
            parts.append(_clean(feature_sets["underwriter"]))
        if "proceeds" in feature_sets:
            parts.append(_clean(feature_sets["proceeds"]))
        if "regime_normalized" in feature_sets:
            parts.append(_clean(feature_sets["regime_normalized"]))
        if parts:
            e2 = parts[0]
            for p in parts[1:]:
                e2 = _merge(e2, p, how="outer")
            v["E2_structured_enhanced"] = e2

        # E3_combined_enhanced: E1 + E2
        if "E1_text_enhanced" in v and "E2_structured_enhanced" in v:
            v["E3_combined_enhanced"] = _merge(
                v["E1_text_enhanced"], v["E2_structured_enhanced"], how="inner"
            )

    return v


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_for_target(
    target: str,
    returns_df: pd.DataFrame,
    variants: dict[str, pd.DataFrame],
    models_dir: Path,
) -> dict:
    """Train all variants × all model types for one target window."""
    if target not in returns_df.columns:
        print(f"  Target '{target}' not in returns.csv — skipping")
        return {}

    target_results: dict = {"target": target, "variants": {}}

    for variant_name, feat_df in variants.items():
        print(f"\n  {'='*52}")
        print(f"  Variant: {variant_name}  |  Target: {target}")

        merged = feat_df.merge(returns_df[["ticker", target]], on="ticker", how="inner")
        merged = merged.dropna(subset=[target])

        if len(merged) < HYPERPARAMS["shared"]["min_samples_to_train"]:
            print(f"  Insufficient data ({len(merged)} samples) — skipping")
            continue

        y = merged[target].values.astype(int)
        feature_cols = [
            c for c in merged.columns
            if c not in ("ticker", target)
            and merged[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        all_nan = [c for c in feature_cols if merged[c].isna().all()]
        if all_nan:
            feature_cols = [c for c in feature_cols if c not in all_nan]

        X = merged[feature_cols].values
        pos_rate = y.mean()
        naive_acc = max(pos_rate, 1 - pos_rate)
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        print(f"  Samples: {len(y)}, Features: {len(feature_cols)}, "
              f"Pos rate: {pos_rate:.1%}, Naive acc: {naive_acc:.1%}")

        variant_results: dict = {}
        print(f"\n  {'Model':<22} {'ROC-AUC':<16} {'Bal-Acc':<16} {'Naive':<8} Pred +%")
        print(f"  {'-'*78}")

        for model_name, model in _build_model_list(scale_pos_weight=scale_pos_weight):
            result = _evaluate_model(model, X, y, feature_cols)
            result["model_type"] = model_name
            variant_results[model_name] = result

            model_path = models_dir / f"{target}_{variant_name}_{model_name}.pkl"
            joblib.dump(model, model_path)

            print(
                f"  {model_name:<22} "
                f"{result['roc_auc_mean']:.3f}±{result['roc_auc_std']:.3f}    "
                f"{result['bal_accuracy_mean']:.3f}±{result['bal_accuracy_std']:.3f}    "
                f"{naive_acc:.3f}   {result['pred_positive_pct']:>5.1f}%"
            )

        target_results["variants"][variant_name] = variant_results

    return target_results


def save_feature_manifest(
    variants: dict[str, pd.DataFrame],
    sources: dict[str, str],
    experiment_dir: Path,
    returns_df: pd.DataFrame,
) -> None:
    """Save feature column list per variant for reproducibility."""
    manifest: dict = {
        "generated_at": datetime.now().isoformat(),
        "feature_sources": sources,
        "variants": {},
    }
    for name, df in variants.items():
        numeric_cols = [
            c for c in df.columns
            if c != "ticker" and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        manifest["variants"][name] = {
            "n_rows": len(df),
            "n_numeric_features": len(numeric_cols),
            "feature_columns": numeric_cols,
        }
    manifest_path = experiment_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Feature manifest saved: {manifest_path}")


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------

def _build_shap_data(
    feat_df: pd.DataFrame,
    target: str,
    returns_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]] | None:
    """Reconstruct the imputed X matrix and feature names for SHAP from a variant DataFrame."""
    from sklearn.impute import SimpleImputer

    merged = feat_df.merge(returns_df[["ticker", target]], on="ticker", how="inner")
    merged = merged.dropna(subset=[target])
    if len(merged) < 10:
        return None

    feature_cols = [
        c for c in merged.columns
        if c not in ("ticker", target)
        and merged[c].dtype in (np.float64, np.float32, np.int64, np.int32)
    ]
    X = SimpleImputer(strategy="median").fit_transform(merged[feature_cols].values)
    return X, feature_cols


def generate_experiment_shap_plots(
    all_results: dict,
    variants: dict[str, pd.DataFrame],
    returns_df: pd.DataFrame,
    models_dir: Path,
    plots_dir: Path,
) -> dict[str, Path]:
    """
    Generate SHAP summary plots for all trained experiment models.
    Mirrors the behaviour of evaluate.py but saves to the experiment's plots/ directory.
    Returns a dict mapping "target/variant/model" → plot Path.
    """
    try:
        from evaluate import generate_shap_plot
    except ImportError:
        # Fall back to absolute import when not running from src/modeling/
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate", Path(__file__).parent / "evaluate.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        generate_shap_plot = mod.generate_shap_plot

    plots_dir.mkdir(parents=True, exist_ok=True)
    all_plot_paths: dict[str, Path] = {}

    for target, target_data in all_results.items():
        for variant_name, models in target_data.get("variants", {}).items():
            feat_df = variants.get(variant_name)
            if feat_df is None:
                continue
            shap_data = _build_shap_data(feat_df, target, returns_df)
            if shap_data is None:
                continue
            X, feature_cols = shap_data

            for model_type in models:
                model_path = models_dir / f"{target}_{variant_name}_{model_type}.pkl"
                if not model_path.exists():
                    continue
                plot_path = generate_shap_plot(
                    model_path, X, feature_cols,
                    f"{target}_{variant_name}", model_type,
                    output_dir=plots_dir,
                )
                if plot_path:
                    all_plot_paths[f"{target}/{variant_name}/{model_type}"] = plot_path
                    print(f"  SHAP saved: {plot_path.name}")

    return all_plot_paths


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(
    targets: list[str],
    which: str,
    experiment_name: str,
    notes: str = "",
) -> tuple[dict, str]:
    """
    Run full training pipeline for an experiment.
    Returns (all_results, run_id).
    """
    run_id    = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()

    # Experiment output directory
    exp_dir      = PROCESSED_DIR / "experiments" / experiment_name
    models_dir   = exp_dir / "models"
    plots_dir    = exp_dir / "plots"
    run_res_dir  = exp_dir / "run_results"
    for d in (exp_dir, models_dir, plots_dir, run_res_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment : {experiment_name}")
    print(f"Run ID     : {run_id}")
    print(f"Variants   : {which}")
    print(f"Targets    : {targets}")
    if notes:
        print(f"Notes      : {notes}")

    returns_df = pd.read_csv(PROCESSED_DIR / "returns.csv")
    feature_sets, sources = load_all_feature_sets()
    variants = build_variants(feature_sets, which=which)

    print(f"\nVariants assembled: {list(variants.keys())}")
    for name, df in variants.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)-1} columns")

    save_feature_manifest(variants, sources, exp_dir, returns_df)

    all_results: dict = {}
    for target in targets:
        print(f"\n{'#'*60}")
        print(f"TARGET: {target}")
        print(f"{'#'*60}")
        result = train_for_target(target, returns_df, variants, models_dir)
        if result:
            all_results[target] = result

    # SHAP plots
    print(f"\n{'='*60}")
    print("Generating SHAP plots...")
    shap_paths = generate_experiment_shap_plots(
        all_results, variants, returns_df, models_dir, plots_dir
    )
    print(f"SHAP plots saved to: {plots_dir}")

    # Save run envelope
    run_envelope = {
        "run_id":           run_id,
        "timestamp":        timestamp,
        "experiment":       experiment_name,
        "notes":            notes,
        "variants_trained": which,
        "targets_trained":  targets,
        "hyperparameters":  HYPERPARAMS,
        "shap_plots":       {k: str(v) for k, v in shap_paths.items()},
        "results":          all_results,
    }
    run_path = run_res_dir / f"{run_id}.json"
    run_path.write_text(json.dumps(run_envelope, indent=2))
    print(f"\nRun saved: {run_path}")

    return all_results, run_id


def print_summary(all_results: dict) -> None:
    for target, res in all_results.items():
        print(f"\n{'='*80}")
        print(f"SUMMARY — {target}")
        print(f"{'='*80}")
        print(f"{'Variant':<28} {'Model':<22} {'ROC-AUC':<18} Bal-Acc")
        print("-" * 80)
        for variant, models in res.get("variants", {}).items():
            for model_name, m in models.items():
                print(
                    f"{variant:<28} {model_name:<22} "
                    f"{m['roc_auc_mean']:.3f}±{m['roc_auc_std']:.3f}   "
                    f"{m['bal_accuracy_mean']:.3f}±{m['bal_accuracy_std']:.3f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPO experiment variants")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help="Experiment name (used as output subdirectory).")
    parser.add_argument("--variants", default="enhanced",
                        choices=["baseline", "enhanced", "all"],
                        help="Which variant family to train.")
    parser.add_argument("--target", default=None,
                        help="Single target (e.g. label_1m). Omit for all targets.")
    parser.add_argument("--notes", default="",
                        help="Optional note attached to the run JSON.")
    args = parser.parse_args()

    targets = [args.target] if args.target else DEFAULT_TARGETS
    results, run_id = run_experiment(
        targets=targets,
        which=args.variants,
        experiment_name=args.experiment,
        notes=args.notes,
    )
    print_summary(results)
    print(f"\nRun ID: {run_id}")
    print(f"Models saved to: data/processed/experiments/{args.experiment}/models/")
