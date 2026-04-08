# Results Tracker

Tracks model performance across runs. Appended automatically by `evaluate.py` after each run.

**Primary metric:** ROC-AUC (0.5 = random chance). Balanced accuracy also reported.  
**Note:** Raw accuracy is misleading for 6m/1y targets due to class imbalance (66–70% naive baseline).

---

## Run 001 — 2026-04-07

### Changes from baseline
- First full run of pipeline

### Setup
| Parameter | Value |
|-----------|-------|
| Filings | ~200 (edgar_fetcher --limit 200) |
| Samples | M1/M3: 425, M2: 427 |
| Targets | label_1m only |
| Models | logistic_regression, xgboost |
| Class balancing | None |
| IPO volume feature | ipos_same_month (contained leakage) |

### Results (label_1m, naive acc: 52.4%)
| Variant | Model | ROC-AUC |
|---------|-------|---------|
| M1_text | logistic_regression | 0.612 ± 0.029 |
| M1_text | xgboost | 0.601 ± 0.030 |
| M2_multiples | logistic_regression | 0.535 ± 0.075 |
| M2_multiples | xgboost | 0.560 ± 0.052 |
| M3_combined | logistic_regression | 0.615 ± 0.038 |
| M3_combined | xgboost | 0.605 ± 0.032 |

### Observations
- M1 (text) outperforms M2 (fundamentals)
- M3 marginally beats M1 (0.615 vs 0.612)
- High std on M2 (±0.075) indicates instability from sparse financial features

---

## Run 002 — 2026-04-07

### Changes from Run 001
- Added Random Forest and Ridge to model suite (4 models total)
- Trained across all 4 return windows (1w, 1m, 6m, 1y)
- Added class balancing: `class_weight="balanced"` on LR + RF; `scale_pos_weight` on XGBoost
- Added balanced accuracy metric and prediction split (% positive/negative predictions)
- Fixed leakage: `ipos_same_month` → `ipos_prior_30d` / `ipos_prior_90d` (strictly trailing)
- M2 now correctly includes market context features alongside financial multiples
- More filings downloaded (full pipeline run)

### Setup
| Parameter | Value |
|-----------|-------|
| Targets | label_1w, label_1m, label_6m, label_1y |
| Models | logistic_regression, ridge, random_forest, xgboost |
| Class balancing | Yes (balanced weights / scale_pos_weight) |
| IPO volume feature | ipos_prior_30d, ipos_prior_90d (leakage-free) |

*Full numeric results below — auto-appended by evaluate.py after next run*

---

---

## Auto-logged — 2026-04-07 23:01


### label_1w  (naive acc: 52.2%)

| Variant | Model | ROC-AUC | Bal-Acc | Pred +% |
|---------|-------|---------|---------|---------|
| M1_text | logistic_regression | 0.516 ± 0.032 | 0.511 ± 0.030 | 52.9% |
| M1_text | ridge | 0.530 ± 0.037 | 0.529 ± 0.053 | 55.8% |
| M1_text | random_forest | 0.607 ± 0.054 | 0.601 ± 0.059 | 56.9% |
| M1_text | xgboost | 0.586 ± 0.033 | 0.573 ± 0.038 | 57.4% |
| M2_multiples | logistic_regression | 0.592 ± 0.045 | 0.575 ± 0.028 | 57.6% |
| M2_multiples | ridge | 0.596 ± 0.057 | 0.565 ± 0.059 | 49.9% |
| M2_multiples | random_forest | 0.608 ± 0.077 | 0.578 ± 0.065 | 54.5% |
| M2_multiples | xgboost | 0.584 ± 0.053 | 0.555 ± 0.059 | 53.8% |
| M3_combined | logistic_regression | 0.510 ± 0.030 | 0.503 ± 0.027 | 55.1% |
| M3_combined | ridge | 0.513 ± 0.060 | 0.521 ± 0.042 | 52.5% |
| M3_combined | random_forest | 0.594 ± 0.055 | 0.585 ± 0.064 | 56.7% |
| M3_combined | xgboost | 0.585 ± 0.063 | 0.564 ± 0.061 | 55.5% |

### label_1m  (naive acc: 50.6%)

| Variant | Model | ROC-AUC | Bal-Acc | Pred +% |
|---------|-------|---------|---------|---------|
| M1_text | logistic_regression | 0.612 ± 0.029 | 0.581 ± 0.044 | 52.2% |
| M1_text | ridge | 0.583 ± 0.052 | 0.558 ± 0.050 | 49.9% |
| M1_text | random_forest | 0.620 ± 0.019 | 0.612 ± 0.032 | 54.4% |
| M1_text | xgboost | 0.607 ± 0.035 | 0.577 ± 0.042 | 56.0% |
| M2_multiples | logistic_regression | 0.566 ± 0.062 | 0.549 ± 0.053 | 52.6% |
| M2_multiples | ridge | 0.553 ± 0.065 | 0.532 ± 0.055 | 45.1% |
| M2_multiples | random_forest | 0.630 ± 0.062 | 0.577 ± 0.050 | 46.4% |
| M2_multiples | xgboost | 0.607 ± 0.071 | 0.571 ± 0.060 | 47.0% |
| M3_combined | logistic_regression | 0.619 ± 0.038 | 0.584 ± 0.042 | 52.5% |
| M3_combined | ridge | 0.587 ± 0.056 | 0.567 ± 0.044 | 50.4% |
| M3_combined | random_forest | 0.631 ± 0.025 | 0.612 ± 0.036 | 53.9% |
| M3_combined | xgboost | 0.615 ± 0.037 | 0.600 ± 0.040 | 53.2% |

### label_6m  (naive acc: 65.9%)

| Variant | Model | ROC-AUC | Bal-Acc | Pred +% |
|---------|-------|---------|---------|---------|
| M1_text | logistic_regression | 0.592 ± 0.022 | 0.551 ± 0.032 | 39.5% |
| M1_text | ridge | 0.558 ± 0.033 | 0.509 ± 0.040 | 40.9% |
| M1_text | random_forest | 0.608 ± 0.044 | 0.549 ± 0.018 | 17.6% |
| M1_text | xgboost | 0.630 ± 0.055 | 0.582 ± 0.044 | 27.8% |
| M2_multiples | logistic_regression | 0.625 ± 0.067 | 0.595 ± 0.047 | 45.5% |
| M2_multiples | ridge | 0.621 ± 0.066 | 0.548 ± 0.009 | 12.6% |
| M2_multiples | random_forest | 0.679 ± 0.057 | 0.634 ± 0.039 | 32.9% |
| M2_multiples | xgboost | 0.637 ± 0.042 | 0.606 ± 0.023 | 37.1% |
| M3_combined | logistic_regression | 0.620 ± 0.035 | 0.569 ± 0.032 | 37.2% |
| M3_combined | ridge | 0.559 ± 0.034 | 0.521 ± 0.040 | 40.7% |
| M3_combined | random_forest | 0.679 ± 0.052 | 0.631 ± 0.021 | 20.0% |
| M3_combined | xgboost | 0.726 ± 0.033 | 0.672 ± 0.042 | 32.5% |

### label_1y  (naive acc: 69.2%)

| Variant | Model | ROC-AUC | Bal-Acc | Pred +% |
|---------|-------|---------|---------|---------|
| M1_text | logistic_regression | 0.644 ± 0.064 | 0.578 ± 0.050 | 35.1% |
| M1_text | ridge | 0.587 ± 0.050 | 0.550 ± 0.037 | 38.8% |
| M1_text | random_forest | 0.649 ± 0.072 | 0.528 ± 0.037 | 10.6% |
| M1_text | xgboost | 0.605 ± 0.051 | 0.553 ± 0.021 | 20.9% |
| M2_multiples | logistic_regression | 0.681 ± 0.060 | 0.641 ± 0.044 | 47.6% |
| M2_multiples | ridge | 0.689 ± 0.059 | 0.581 ± 0.032 | 14.9% |
| M2_multiples | random_forest | 0.690 ± 0.042 | 0.633 ± 0.023 | 34.0% |
| M2_multiples | xgboost | 0.681 ± 0.030 | 0.656 ± 0.045 | 36.6% |
| M3_combined | logistic_regression | 0.689 ± 0.073 | 0.605 ± 0.055 | 35.8% |
| M3_combined | ridge | 0.606 ± 0.051 | 0.574 ± 0.044 | 36.9% |
| M3_combined | random_forest | 0.688 ± 0.084 | 0.559 ± 0.042 | 12.5% |
| M3_combined | xgboost | 0.671 ± 0.039 | 0.629 ± 0.035 | 29.4% |
