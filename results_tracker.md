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
