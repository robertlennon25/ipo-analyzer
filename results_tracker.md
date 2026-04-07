# Results Tracker

Tracks model performance across runs. Target is always binary 1-month return (`label_1m`) unless noted.
ROC-AUC is the primary metric (0.5 = random, higher = better). Std is from 5-fold stratified CV.

---

## Run 001 — 2026-04-07

### Setup
| Parameter | Value |
|-----------|-------|
| Date | 2026-04-07 |
| Target | `label_1m` (1-month return > 0) |
| IPO universe | 700 scraped (2019–2023), 515–517 with price data |
| Filings downloaded | ~200 (edgar_fetcher --limit 200) |
| Samples in model | M1/M3: 425, M2: 427 |
| CV folds | 5 (StratifiedKFold) |
| Sector coverage | 515 / 700 with known sector (185 Unknown) |

### Feature counts
| Variant | Features | Description |
|---------|----------|-------------|
| M1_text | 401 | Handcrafted NLP (VADER sentiment, keyword densities, readability) + 384-dim `all-MiniLM-L6-v2` embeddings |
| M2_multiples | 18 | Financial features (revenue, margins, proceeds) + market context (VIX, S&P momentum, sector ETF) |
| M3_combined | 419 | M1 + M2 combined |

### Results
| Variant | Model | ROC-AUC | Accuracy |
|---------|-------|---------|----------|
| M1_text | logistic_regression | 0.612 ± 0.029 | 0.586 ± 0.044 |
| M1_text | xgboost | 0.601 ± 0.030 | 0.576 ± 0.025 |
| M2_multiples | logistic_regression | 0.535 ± 0.075 | 0.529 ± 0.080 |
| M2_multiples | xgboost | 0.560 ± 0.052 | 0.567 ± 0.050 |
| M3_combined | logistic_regression | 0.615 ± 0.038 | 0.584 ± 0.033 |
| M3_combined | xgboost | 0.605 ± 0.032 | 0.560 ± 0.042 |

### Observations
- M1 (text only) outperforms M2 (fundamentals only) — text features carry more signal at this sample size
- M3 marginally beats M1 alone (0.615 vs 0.612 for LR), suggesting financials add a small increment over text
- M2 XGBoost (0.560) now above random after fixing all-NaN column drop — but high std (±0.075 for LR) indicates instability
- All results are weak signal territory (0.5–0.65); 425 samples is at the low end for reliable conclusions
- SHAP plots saved to `data/processed/plots/`

### Known limitations this run
- ~185 / 700 IPOs have Unknown sector (affects sector ETF features in M3)
- `total_proceeds_m` is a placeholder (100) for most IPOs
- Financial feature coverage is sparse — many filings had extraction failures in `multiples.py`
- Sample size (~425) too small for definitive conclusions; target ≥ 300 clean filings for next run

---
