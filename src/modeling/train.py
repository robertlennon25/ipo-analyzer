"""
train.py
--------
Train and compare M1 / M2 / M3 model variants.

M1: Text only (embeddings + handcrafted NLP features)
M2: Multiples + deal structure only (structured financial features)
M3: M1 + M2 combined

Each variant is trained with:
- Logistic Regression (interpretability baseline)
- XGBoost (performance)

Evaluation:
- ROC-AUC (primary)
- Accuracy, Precision, Recall, F1
- Cross-validated scores
- Feature importance

Output: data/processed/model_results.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CACHE_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS

MODELS_DIR = PROCESSED_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = PROCESSED_DIR / "model_results.json"

# Primary target
DEFAULT_TARGET = "label_1m"  # 1-month binary return


def load_feature_sets() -> dict[str, pd.DataFrame]:
    """Load all feature sets and return as dict."""
    from src.features.embeddings import load_embeddings, embeddings_to_dataframe

    feature_sets = {}

    # Handcrafted text features
    hc_path = PROCESSED_DIR / "handcrafted_features.csv"
    if hc_path.exists():
        feature_sets["handcrafted"] = pd.read_csv(hc_path)

    # Multiples / deal structure
    multiples_path = PROCESSED_DIR / "multiples_features.csv"
    if multiples_path.exists():
        feature_sets["multiples"] = pd.read_csv(multiples_path)

    # Embeddings
    try:
        embeddings, tickers = load_embeddings()
        feature_sets["embeddings"] = embeddings_to_dataframe(embeddings, tickers)
    except FileNotFoundError:
        print("Embeddings not found — skipping")

    return feature_sets


def build_model_variants(feature_sets: dict, returns_df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """
    Assemble feature matrices for each model variant.
    Returns dict: {variant_name: feature_df}
    """
    variants = {}

    base_cols = ["ticker"]

    # Text features (handcrafted + embeddings)
    text_parts = []
    if "handcrafted" in feature_sets:
        hc = feature_sets["handcrafted"].copy()
        drop_cols = ["ticker", "filing_type"] + [c for c in hc.columns if hc[c].dtype == object]
        text_parts.append(hc.drop(columns=[c for c in drop_cols if c in hc.columns and c != "ticker"]))
    if "embeddings" in feature_sets:
        text_parts.append(feature_sets["embeddings"])

    if text_parts:
        m1 = text_parts[0]
        for part in text_parts[1:]:
            m1 = m1.merge(part, on="ticker", how="inner")
        variants["M1_text"] = m1

    # Structured features (multiples + deal metadata)
    if "multiples" in feature_sets:
        variants["M2_multiples"] = feature_sets["multiples"].copy()

    # Combined
    if "M1_text" in variants and "M2_multiples" in variants:
        m3 = variants["M1_text"].merge(variants["M2_multiples"], on="ticker", how="inner")
        variants["M3_combined"] = m3

    return variants


def evaluate_model(model, X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Cross-validate and return evaluation metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Fit on full data for feature importance
    model.fit(X, y)

    result = {
        "roc_auc_mean": float(auc_scores.mean()),
        "roc_auc_std": float(auc_scores.std()),
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
    }

    # Feature importance (XGBoost or LR coefficients)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "named_steps"):
        # Pipeline
        final = list(model.named_steps.values())[-1]
        if hasattr(final, "feature_importances_"):
            importance = final.feature_importances_
        elif hasattr(final, "coef_"):
            importance = np.abs(final.coef_[0])
        else:
            importance = None
    else:
        importance = None

    if importance is not None and len(feature_names) == len(importance):
        top_idx = np.argsort(importance)[::-1][:20]
        result["top_features"] = [
            {"feature": feature_names[i], "importance": float(importance[i])}
            for i in top_idx
        ]

    return result


def train_all_variants(target: str = DEFAULT_TARGET) -> dict:
    """Main training loop. Trains all variants × all model types."""
    returns_df = pd.read_csv(PROCESSED_DIR / "returns.csv")

    if target not in returns_df.columns:
        available = [c for c in returns_df.columns if c.startswith("label_")]
        print(f"Target '{target}' not found. Available: {available}")
        target = available[0] if available else None

    if not target:
        raise ValueError("No valid target column found in returns.csv")

    feature_sets = load_feature_sets()
    variants = build_model_variants(feature_sets, returns_df, target)

    all_results = {"target": target, "variants": {}}

    for variant_name, feat_df in variants.items():
        print(f"\n{'='*50}")
        print(f"Training variant: {variant_name}")

        # Merge with target
        merged = feat_df.merge(returns_df[["ticker", target]], on="ticker", how="inner")
        merged = merged.dropna(subset=[target])

        if len(merged) < 30:
            print(f"  Insufficient data ({len(merged)} samples), skipping")
            continue

        y = merged[target].values.astype(int)
        feature_cols = [c for c in merged.columns if c not in ["ticker", target]
                        and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        # Drop columns that are entirely NaN — imputer can't handle them
        all_nan = [c for c in feature_cols if merged[c].isna().all()]
        if all_nan:
            print(f"  Dropping {len(all_nan)} all-NaN columns: {all_nan}")
            feature_cols = [c for c in feature_cols if c not in all_nan]

        X = merged[feature_cols].values

        print(f"  Samples: {len(y)}, Features: {len(feature_cols)}, Positive rate: {y.mean():.1%}")

        variant_results = {}

        for model_name, model in [
            ("logistic_regression", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.1)),
            ])),
            ("xgboost", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    random_state=RANDOM_STATE, eval_metric="logloss",
                    verbosity=0,
                )),
            ])),
        ]:
            print(f"  → {model_name}")
            result = evaluate_model(model, X, y, feature_cols)
            result["model_type"] = model_name
            variant_results[model_name] = result

            # Save trained model
            joblib.dump(model, MODELS_DIR / f"{variant_name}_{model_name}.pkl")
            print(f"    ROC-AUC: {result['roc_auc_mean']:.3f} ± {result['roc_auc_std']:.3f}")
            print(f"    Accuracy: {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")

        all_results["variants"][variant_name] = variant_results

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return all_results


def print_comparison_table(results: dict):
    """Print a clean comparison table of model variants."""
    print("\n" + "="*70)
    print(f"MODEL COMPARISON (target: {results['target']})")
    print("="*70)
    print(f"{'Variant':<20} {'Model':<25} {'ROC-AUC':<12} {'Accuracy':<10}")
    print("-"*70)

    for variant, models in results["variants"].items():
        for model_name, metrics in models.items():
            print(
                f"{variant:<20} {model_name:<25} "
                f"{metrics['roc_auc_mean']:.3f}±{metrics['roc_auc_std']:.3f}   "
                f"{metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}"
            )


if __name__ == "__main__":
    results = train_all_variants()
    print_comparison_table(results)
