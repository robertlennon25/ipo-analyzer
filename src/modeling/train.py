"""
train.py
--------
Train and compare M1 / M2 / M3 model variants across multiple return windows.

Model variants:
  M1: Text only (embeddings + handcrafted NLP features)
  M2: Multiples + deal structure + market context (structured features)
  M3: M1 + M2 combined

Model types per variant:
  - Logistic Regression   (interpretability baseline)
  - Ridge Classifier      (L2-regularized linear)
  - Random Forest         (non-linear, low variance)
  - XGBoost               (performance)

Default: trains all variants × all model types × all return windows (1w, 1m, 6m, 1y).
Pass --target label_1m to train a single window only.
Pass --notes "..." to attach a note to the saved run file.

Output: data/processed/run_results/{run_id}.json   ← timestamped, never overwritten
        data/processed/model_results.json           ← latest run (overwritten each time)
        data/processed/models/{target}_{variant}_{model}.pkl
"""

import json
import argparse
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
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CACHE_DIR, RANDOM_STATE, CV_FOLDS

MODELS_DIR = PROCESSED_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
RUN_RESULTS_DIR = PROCESSED_DIR / "run_results"
RUN_RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = PROCESSED_DIR / "model_results.json"

DEFAULT_TARGETS = ["label_1w", "label_1m", "label_6m", "label_1y"]

# Hyperparameters — single source of truth, serialised into every run JSON.
HYPERPARAMS = {
    "logistic_regression": {
        "C": 0.1, "max_iter": 1000, "class_weight": "balanced",
    },
    "ridge": {
        "alpha": 1.0,
        "note": "no class_weight — interpret with caution on imbalanced targets",
    },
    "random_forest": {
        "n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    "xgboost": {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
        "eval_metric": "logloss",
        "note": "scale_pos_weight computed per-target from class ratio",
    },
    "shared": {
        "imputer_strategy": "median",
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "min_samples_to_train": 30,
    },
}


def _build_model_list(scale_pos_weight: float = 1.0) -> list[tuple[str, Pipeline]]:
    """
    scale_pos_weight = n_negative / n_positive — passed to XGBoost to handle class imbalance.
    LR and RF use class_weight='balanced' which computes this automatically.
    Ridge has no class_weight support; it will be biased toward the majority class.
    """
    hp = HYPERPARAMS
    return [
        ("logistic_regression", Pipeline([
            ("imputer", SimpleImputer(strategy=hp["shared"]["imputer_strategy"])),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=hp["logistic_regression"]["max_iter"],
                random_state=RANDOM_STATE,
                C=hp["logistic_regression"]["C"],
                class_weight=hp["logistic_regression"]["class_weight"],
            )),
        ])),
        ("ridge", Pipeline([
            ("imputer", SimpleImputer(strategy=hp["shared"]["imputer_strategy"])),
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=hp["ridge"]["alpha"])),
        ])),
        ("random_forest", Pipeline([
            ("imputer", SimpleImputer(strategy=hp["shared"]["imputer_strategy"])),
            ("clf", RandomForestClassifier(
                n_estimators=hp["random_forest"]["n_estimators"],
                max_depth=hp["random_forest"]["max_depth"],
                min_samples_leaf=hp["random_forest"]["min_samples_leaf"],
                random_state=RANDOM_STATE, n_jobs=-1,
                class_weight=hp["random_forest"]["class_weight"],
            )),
        ])),
        ("xgboost", Pipeline([
            ("imputer", SimpleImputer(strategy=hp["shared"]["imputer_strategy"])),
            ("clf", xgb.XGBClassifier(
                n_estimators=hp["xgboost"]["n_estimators"],
                max_depth=hp["xgboost"]["max_depth"],
                learning_rate=hp["xgboost"]["learning_rate"],
                random_state=RANDOM_STATE,
                eval_metric=hp["xgboost"]["eval_metric"],
                verbosity=0,
                scale_pos_weight=scale_pos_weight,
            )),
        ])),
    ]


def load_feature_sets() -> dict[str, pd.DataFrame]:
    """Load all feature sets and return as dict."""
    from src.features.embeddings import load_embeddings, embeddings_to_dataframe

    feature_sets = {}

    hc_path = PROCESSED_DIR / "handcrafted_features.csv"
    if hc_path.exists():
        feature_sets["handcrafted"] = pd.read_csv(hc_path)

    multiples_path = PROCESSED_DIR / "multiples_features.csv"
    if multiples_path.exists():
        feature_sets["multiples"] = pd.read_csv(multiples_path)

    market_path = PROCESSED_DIR / "market_context_features.csv"
    if market_path.exists():
        feature_sets["market"] = pd.read_csv(market_path)

    try:
        embeddings, tickers = load_embeddings()
        feature_sets["embeddings"] = embeddings_to_dataframe(embeddings, tickers)
    except FileNotFoundError:
        print("Embeddings not found — skipping")

    return feature_sets


def build_model_variants(feature_sets: dict) -> dict[str, pd.DataFrame]:
    """
    Assemble feature matrices for each model variant.
    Returns dict: {variant_name: feature_df}
    """
    variants = {}

    # M1: text features (handcrafted + embeddings)
    text_parts = []
    if "handcrafted" in feature_sets:
        hc = feature_sets["handcrafted"].copy()
        drop_obj = [c for c in hc.columns if hc[c].dtype == object and c != "ticker"]
        text_parts.append(hc.drop(columns=drop_obj))
    if "embeddings" in feature_sets:
        text_parts.append(feature_sets["embeddings"])

    if text_parts:
        m1 = text_parts[0]
        for part in text_parts[1:]:
            m1 = m1.merge(part, on="ticker", how="inner")
        variants["M1_text"] = m1

    # M2: structured features (multiples + market context)
    struct_parts = []
    if "multiples" in feature_sets:
        mult = feature_sets["multiples"].copy()
        drop_obj = [c for c in mult.columns if mult[c].dtype == object and c != "ticker"]
        struct_parts.append(mult.drop(columns=drop_obj))
    if "market" in feature_sets:
        mkt = feature_sets["market"].copy()
        drop_obj = [c for c in mkt.columns if mkt[c].dtype == object and c != "ticker"]
        struct_parts.append(mkt.drop(columns=drop_obj))

    if struct_parts:
        m2 = struct_parts[0]
        for part in struct_parts[1:]:
            m2 = m2.merge(part, on="ticker", how="outer")
        variants["M2_multiples"] = m2

    # M3: combined
    if "M1_text" in variants and "M2_multiples" in variants:
        variants["M3_combined"] = variants["M1_text"].merge(
            variants["M2_multiples"], on="ticker", how="inner"
        )

    return variants


def evaluate_model(model, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Cross-validate and return evaluation metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # RidgeClassifier has no predict_proba — use decision_function for ROC-AUC
    clf = list(model.named_steps.values())[-1]
    if hasattr(clf, "predict_proba"):
        scoring = "roc_auc"
    else:
        scoring = "roc_auc"  # sklearn uses decision_function automatically

    auc_scores     = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    acc_scores     = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    bal_acc_scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
    oof_preds      = cross_val_predict(model, X, y, cv=cv)

    pred_pos_rate = float(oof_preds.mean())

    model.fit(X, y)

    result = {
        "roc_auc_mean":      float(auc_scores.mean()),
        "roc_auc_std":       float(auc_scores.std()),
        "accuracy_mean":     float(acc_scores.mean()),
        "accuracy_std":      float(acc_scores.std()),
        "bal_accuracy_mean": float(bal_acc_scores.mean()),
        "bal_accuracy_std":  float(bal_acc_scores.std()),
        "pred_positive_pct": round(pred_pos_rate * 100, 1),
        "pred_negative_pct": round((1 - pred_pos_rate) * 100, 1),
        "n_samples":         int(len(y)),
        "n_features":        int(X.shape[1]),
        "positive_rate":     float(y.mean()),
        "naive_accuracy":    float(max(y.mean(), 1 - y.mean())),
    }

    # Feature importance
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


def train_for_target(
    target: str,
    returns_df: pd.DataFrame,
    variants: dict[str, pd.DataFrame],
) -> dict:
    """Train all variants × all model types for one target window."""
    if target not in returns_df.columns:
        print(f"  Target '{target}' not in returns.csv — skipping")
        return {}

    target_results = {"target": target, "variants": {}}

    for variant_name, feat_df in variants.items():
        print(f"\n  {'='*50}")
        print(f"  Variant: {variant_name}  |  Target: {target}")

        merged = feat_df.merge(returns_df[["ticker", target]], on="ticker", how="inner")
        merged = merged.dropna(subset=[target])

        if len(merged) < 30:
            print(f"  Insufficient data ({len(merged)} samples) — skipping")
            continue

        y = merged[target].values.astype(int)
        feature_cols = [
            c for c in merged.columns
            if c not in ["ticker", target]
            and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        all_nan = [c for c in feature_cols if merged[c].isna().all()]
        if all_nan:
            print(f"  Dropping {len(all_nan)} all-NaN columns")
            feature_cols = [c for c in feature_cols if c not in all_nan]

        X = merged[feature_cols].values
        pos_rate = y.mean()
        naive_acc = max(pos_rate, 1 - pos_rate)
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        print(f"  Samples: {len(y)}, Features: {len(feature_cols)}, "
              f"Positive rate: {pos_rate:.1%}, Naive acc: {naive_acc:.1%}")

        variant_results = {}
        print(f"\n  {'Model':<22} {'ROC-AUC':<16} {'Bal-Acc':<16} {'Naive':<8} {'Pred +%':<9} Pred -%")
        print(f"  {'-'*78}")
        for model_name, model in _build_model_list(scale_pos_weight=scale_pos_weight):
            result = evaluate_model(model, X, y, feature_cols)
            result["model_type"] = model_name
            variant_results[model_name] = result
            joblib.dump(model, MODELS_DIR / f"{target}_{variant_name}_{model_name}.pkl")
            print(
                f"  {model_name:<22} "
                f"{result['roc_auc_mean']:.3f}±{result['roc_auc_std']:.3f}    "
                f"{result['bal_accuracy_mean']:.3f}±{result['bal_accuracy_std']:.3f}    "
                f"{naive_acc:.3f}   "
                f"{result['pred_positive_pct']:>5.1f}%    "
                f"{result['pred_negative_pct']:>5.1f}%"
            )

        target_results["variants"][variant_name] = variant_results

    return target_results


def train_all(targets: list[str], notes: str = "") -> tuple[dict, str]:
    """
    Main entry point. Trains all variants × models × targets.
    Returns (all_results, run_id).
    """
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat()

    returns_df = pd.read_csv(PROCESSED_DIR / "returns.csv")
    feature_sets = load_feature_sets()
    variants = build_model_variants(feature_sets)

    all_results = {}
    for target in targets:
        print(f"\n{'#'*60}")
        print(f"TARGET: {target}")
        print(f"{'#'*60}")
        result = train_for_target(target, returns_df, variants)
        if result:
            all_results[target] = result

    # Write legacy model_results.json (overwritten each run — used by evaluate.py fallback)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    # Write timestamped run file with full metadata envelope
    run_envelope = {
        "run_id": run_id,
        "timestamp": timestamp,
        "notes": notes,
        "targets_trained": targets,
        "hyperparameters": HYPERPARAMS,
        "results": all_results,
    }
    run_path = RUN_RESULTS_DIR / f"{run_id}.json"
    with open(run_path, "w") as f:
        json.dump(run_envelope, f, indent=2)

    print(f"\nRun saved: {run_path}")
    print(f"Latest results: {RESULTS_PATH}")

    return all_results, run_id


def print_comparison_table(all_results: dict):
    """Print a comparison table for each target window."""
    for target, results in all_results.items():
        # Get naive accuracy from first result that has it
        naive = None
        for models in results.get("variants", {}).values():
            for metrics in models.values():
                naive = metrics.get("naive_accuracy")
                break
            if naive:
                break

        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON  (target: {target}"
              + (f", naive acc: {naive:.1%}" if naive else "") + ")")
        print(f"{'='*80}")
        print(f"{'Variant':<20} {'Model':<22} {'ROC-AUC':<18} {'Bal-Accuracy'}")
        print("-" * 80)
        for variant, models in results.get("variants", {}).items():
            for model_name, metrics in models.items():
                print(
                    f"{variant:<20} {model_name:<22} "
                    f"{metrics['roc_auc_mean']:.3f}±{metrics['roc_auc_std']:.3f}   "
                    f"{metrics['bal_accuracy_mean']:.3f}±{metrics['bal_accuracy_std']:.3f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPO model variants")
    parser.add_argument(
        "--target", type=str, default=None,
        help="Single target to train (e.g. label_1m). Omit to train all windows."
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Optional note to attach to this run (e.g. 'after leakage fix, 300 filings')."
    )
    args = parser.parse_args()

    targets = [args.target] if args.target else DEFAULT_TARGETS
    print(f"Training targets: {targets}")
    if args.notes:
        print(f"Notes: {args.notes}")

    results, run_id = train_all(targets, notes=args.notes)
    print_comparison_table(results)
    print(f"\nRun ID: {run_id}")
