"""
temporal_split_test.py
----------------------
Leakage sanity check #2: temporal train/test split.

Trains on the chronologically earlier half of IPOs and evaluates on the later
half. This catches two things:

1. Year-level confounding — if is_hot_ipo_year / ipo_year are doing the
   heavy lifting, the model trained on 2019-2021 will fail on 2022-2023
   because it learned year-specific patterns, not generalizable signals.

2. Temporal stability — a genuinely predictive model should retain some
   AUC on held-out future IPOs. If AUC collapses to ~0.5 out-of-sample
   this suggests overfitting to the training period.

Unlike train.py (which uses cross-validation), this is a strict one-way
split: no future data ever leaks into training.

Usage:
    python src/modeling-test-leakage/temporal_split_test.py
    python src/modeling-test-leakage/temporal_split_test.py --target label_1m --split 0.5
"""

import argparse
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))

from config.settings import PROCESSED_DIR, RANDOM_STATE
from src.modeling.train import (
    load_feature_sets,
    build_model_variants,
    _build_model_list,
)

RESULTS_DIR = PROCESSED_DIR / "leakage-test-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TARGET = "label_1m"
DEFAULT_SPLIT = 0.5   # fraction of data used for training (by date order)


def run_temporal_split_test(target: str, split_frac: float) -> None:
    returns_df = pd.read_csv(PROCESSED_DIR / "returns.csv", parse_dates=["ipo_date"])
    if target not in returns_df.columns:
        print(f"Target '{target}' not in returns.csv")
        sys.exit(1)

    # Sort by IPO date — this is the key difference from train.py
    returns_df = returns_df.sort_values("ipo_date").reset_index(drop=True)

    feature_sets = load_feature_sets()
    variants = build_model_variants(feature_sets)

    cutoff_idx = int(len(returns_df) * split_frac)
    cutoff_date = returns_df.iloc[cutoff_idx]["ipo_date"]
    train_dates = returns_df[returns_df["ipo_date"] <  cutoff_date]["ipo_date"]
    test_dates  = returns_df[returns_df["ipo_date"] >= cutoff_date]["ipo_date"]

    print(f"\nTemporal split test — target: {target}")
    print(f"Split point: {str(cutoff_date)[:10]}  ({split_frac:.0%} train / {1-split_frac:.0%} test)")
    print(f"Train period: {str(train_dates.min())[:10]} → {str(train_dates.max())[:10]}  (n={len(train_dates)})")
    print(f"Test  period: {str(test_dates.min())[:10]}  → {str(test_dates.max())[:10]}  (n={len(test_dates)})\n")

    output: dict = {
        "test": "temporal_split",
        "target": target,
        "split_frac": split_frac,
        "cutoff_date": str(cutoff_date)[:10],
        "train_period": f"{str(train_dates.min())[:10]} → {str(train_dates.max())[:10]}",
        "test_period":  f"{str(test_dates.min())[:10]} → {str(test_dates.max())[:10]}",
        "run_at": datetime.now().isoformat(),
        "variants": {},
    }

    train_tickers = set(returns_df[returns_df["ipo_date"] <  cutoff_date]["ticker"])
    test_tickers  = set(returns_df[returns_df["ipo_date"] >= cutoff_date]["ticker"])

    for variant_name, feat_df in variants.items():
        merged = feat_df.merge(returns_df[["ticker", target, "ipo_date"]], on="ticker", how="inner")
        merged = merged.dropna(subset=[target])

        feature_cols = [
            c for c in merged.columns
            if c not in ["ticker", target, "ipo_date"]
            and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not merged[c].isna().all()
        ]

        train_df = merged[merged["ticker"].isin(train_tickers)]
        test_df  = merged[merged["ticker"].isin(test_tickers)]

        if len(train_df) < 20 or len(test_df) < 10:
            print(f"{variant_name}: insufficient data (train={len(train_df)}, test={len(test_df)}) — skipping\n")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values.astype(int)
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target].values.astype(int)

        # Fit imputer on train only — do NOT fit on test data
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        train_pos = y_train.mean()
        test_pos  = y_test.mean()
        scale_pos_weight = (1 - train_pos) / train_pos if train_pos > 0 else 1.0

        print(f"{'='*65}")
        print(f"Variant: {variant_name}")
        print(f"  Train: n={len(y_train)}, pos={train_pos:.1%}  |  Test: n={len(y_test)}, pos={test_pos:.1%}")
        print(f"  Features: {len(feature_cols)}")
        print(f"{'='*65}")
        print(f"{'Model':<22} {'Train AUC':>11} {'Test AUC':>10} {'Drop':>8} {'Verdict'}")
        print(f"{'-'*65}")

        model_rows = {}
        for model_name, model in _build_model_list(scale_pos_weight=scale_pos_weight):
            try:
                model.fit(X_train, y_train)

                clf = list(model.named_steps.values())[-1]
                if hasattr(clf, "predict_proba"):
                    train_proba = model.predict_proba(X_train)[:, 1]
                    test_proba  = model.predict_proba(X_test)[:, 1]
                else:
                    train_proba = model.decision_function(X_train)
                    test_proba  = model.decision_function(X_test)

                train_auc = float(roc_auc_score(y_train, train_proba))
                test_auc  = float(roc_auc_score(y_test,  test_proba))
                drop = train_auc - test_auc

                if test_auc >= 0.55:
                    verdict = "GENERALIZES"
                elif test_auc >= 0.50:
                    verdict = "MARGINAL"
                else:
                    verdict = "COLLAPSES (<0.5)"

                print(
                    f"{model_name:<22} {train_auc:>11.3f} {test_auc:>10.3f} "
                    f"{drop:>+8.3f}  {verdict}"
                )
                model_rows[model_name] = {
                    "train_auc": round(train_auc, 4),
                    "test_auc":  round(test_auc, 4),
                    "drop":      round(drop, 4),
                    "verdict":   verdict,
                }
            except Exception as e:
                print(f"{model_name:<22} ERROR: {e}")
                model_rows[model_name] = {"error": str(e)}

        output["variants"][variant_name] = {
            "n_train": int(len(y_train)),
            "n_test":  int(len(y_test)),
            "n_features": int(len(feature_cols)),
            "train_positive_rate": round(float(train_pos), 4),
            "test_positive_rate":  round(float(test_pos), 4),
            "models": model_rows,
        }
        print()

    print("Interpretation:")
    print("  Test AUC ≥ 0.55 consistently          → signal generalises across time")
    print("  Train AUC >> test AUC (large drop)     → overfitting or year-level confounding")
    print("  Test AUC < 0.50                        → model learned training-period-specific patterns")
    print("  is_hot_ipo_year/ipo_year confounding:  → if M2/M3 drops sharply, year features are driving results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"temporal_{target}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal split leakage test")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="Target column (e.g. label_6m). Default: label_1m")
    parser.add_argument("--split", type=float, default=DEFAULT_SPLIT,
                        help="Fraction of data (by date) used for training. Default: 0.5")
    args = parser.parse_args()
    run_temporal_split_test(args.target, args.split)
