"""
permutation_test.py
-------------------
Leakage sanity check #1: label permutation test.

Trains the same models as train.py, but replaces the real labels with a
randomly shuffled version. If the model has no leakage, AUC should collapse
to ~0.5 on shuffled labels. If AUC stays elevated, features contain
post-IPO information.

Runs N_SHUFFLES independent permutations to build a null distribution,
then compares each model's real AUC against the null.

Usage:
    python src/modeling-test-leakage/permutation_test.py
    python src/modeling-test-leakage/permutation_test.py --target label_1m --shuffles 50
"""

import argparse
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from config.settings import PROCESSED_DIR, RANDOM_STATE, CV_FOLDS
from src.modeling.train import (
    load_feature_sets,
    build_model_variants,
    _build_model_list,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer

RESULTS_DIR = PROCESSED_DIR / "leakage-test-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SHUFFLES_DEFAULT = 20
DEFAULT_TARGET = "label_1m"


def auc_for_model(model, X: np.ndarray, y: np.ndarray) -> float:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def run_permutation_test(target: str, n_shuffles: int) -> None:
    returns_df = pd.read_csv(PROCESSED_DIR / "returns.csv")
    if target not in returns_df.columns:
        print(f"Target '{target}' not in returns.csv")
        sys.exit(1)

    feature_sets = load_feature_sets()
    variants = build_model_variants(feature_sets)

    print(f"\nPermutation test — target: {target}, shuffles: {n_shuffles}")
    print(f"Expected null AUC: ~0.500  (any model scoring >> 0.5 on shuffled labels = leakage)\n")

    rng = np.random.default_rng(RANDOM_STATE)
    output: dict = {
        "test": "permutation",
        "target": target,
        "n_shuffles": n_shuffles,
        "run_at": datetime.now().isoformat(),
        "variants": {},
    }

    for variant_name, feat_df in variants.items():
        merged = feat_df.merge(returns_df[["ticker", target]], on="ticker", how="inner")
        merged = merged.dropna(subset=[target])
        if len(merged) < 30:
            print(f"{variant_name}: insufficient data ({len(merged)} samples) — skipping\n")
            continue

        y_real = merged[target].values.astype(int)
        feature_cols = [
            c for c in merged.columns
            if c not in ["ticker", target]
            and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not merged[c].isna().all()
        ]
        X_raw = merged[feature_cols].values

        # Impute once — same imputed X used for all shuffles
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_raw)

        pos_rate = y_real.mean()
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        print(f"{'='*60}")
        print(f"Variant: {variant_name}  |  n={len(y_real)}, features={len(feature_cols)}, pos={pos_rate:.1%}")
        print(f"{'='*60}")
        print(f"{'Model':<22} {'Real AUC':>10} {'Null mean':>10} {'Null std':>9} {'p-value':>9} {'Verdict'}")
        print(f"{'-'*75}")

        variant_rows = {}
        for model_name, model in _build_model_list(scale_pos_weight=scale_pos_weight):
            # Real AUC (on actual labels)
            real_auc = auc_for_model(model, X, y_real)

            # Null distribution: AUC on N_SHUFFLES permutations of labels
            null_aucs = []
            for _ in range(n_shuffles):
                y_shuffled = rng.permutation(y_real)
                try:
                    null_auc = auc_for_model(model, X, y_shuffled)
                    null_aucs.append(null_auc)
                except Exception:
                    pass

            null_mean = float(np.mean(null_aucs))
            null_std  = float(np.std(null_aucs))
            p_value   = float(np.mean([a >= real_auc for a in null_aucs]))

            if p_value < 0.05:
                verdict = "SIGNAL (p<0.05)"
            elif real_auc - null_mean < 0.02:
                verdict = "NO SIGNAL"
            else:
                verdict = "MARGINAL"

            print(
                f"{model_name:<22} {real_auc:>10.3f} {null_mean:>10.3f} "
                f"{null_std:>9.3f} {p_value:>9.3f}  {verdict}"
            )
            variant_rows[model_name] = {
                "real_auc": round(real_auc, 4),
                "null_auc_mean": round(null_mean, 4),
                "null_auc_std":  round(null_std, 4),
                "p_value":       round(p_value, 4),
                "verdict":       verdict,
            }

        output["variants"][variant_name] = {
            "n_samples": int(len(y_real)),
            "n_features": int(len(feature_cols)),
            "positive_rate": round(float(pos_rate), 4),
            "models": variant_rows,
        }
        print()

    print("Interpretation:")
    print("  Real AUC >> null mean + consistent p < 0.05 → genuine signal, no leakage")
    print("  Real AUC ≈ null mean                        → model learns nothing / possible leakage")
    print("  Real AUC >> 0.5 AND null mean >> 0.5        → LEAKAGE — features contain post-IPO info")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"permutation_{target}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation leakage test")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="Target column (e.g. label_1m). Default: label_1m")
    parser.add_argument("--shuffles", type=int, default=N_SHUFFLES_DEFAULT,
                        help=f"Number of label permutations. Default: {N_SHUFFLES_DEFAULT}")
    args = parser.parse_args()
    run_permutation_test(args.target, args.shuffles)
