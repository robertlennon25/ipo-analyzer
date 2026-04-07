# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does
Research system testing whether language in IPO filings (SEC S-1 / 424B4) predicts post-IPO stock performance. Primary question: **does text signal add alpha over fundamentals?**

Three model variants trained across four return windows:
- **M1** — text only (handcrafted NLP + `all-MiniLM-L6-v2` sentence-transformer embeddings)
- **M2** — structured features (financial multiples + market context: VIX, S&P, sector ETF, IPO volume)
- **M3** — M1 + M2 combined

Four model types per variant: Logistic Regression, Ridge, Random Forest, XGBoost.
Four targets: `label_1w`, `label_1m`, `label_6m`, `label_1y` (binary: return > 0).

---

## Setup & Commands

### Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers   # not in requirements.txt; needed for embeddings.py
```

### Full pipeline (run in this order)

```bash
# 1. Build IPO universe with sectors (~4 min for 700 tickers via yfinance)
python src/ingestion/scrape_ipo_universe.py   # scrapes stockanalysis.com + fetches sectors
python src/ingestion/ipo_list.py              # filters → data/processed/ipo_universe.csv

# 2. Fetch price returns (fast, ~5 min, cached)
python src/ingestion/price_fetcher.py

# 3. Download filings from EDGAR (slow — rate-limited ~10 req/s)
python src/ingestion/edgar_fetcher.py --limit 50   # test; drop --limit for full ~700 (3-5 hrs)

# 4. Market context features (no filing dependency — can run alongside step 3)
python src/features/market_context.py

# 5. Extract text sections from filings (skips already-processed tickers)
python src/parsing/section_extractor.py

# 6. Compute features
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py

# 7. Train all variants × models × windows (~10-15 min)
python src/modeling/train.py                  # trains all 4 targets by default
python src/modeling/train.py --target label_1m  # single target only

# 8. Evaluate: SHAP plots + report (auto-appends to results_tracker.md)
python src/modeling/evaluate.py
```

### Re-running after downloading more filings
Steps 1, 2, 4 don't need to re-run. Only:
```bash
python src/ingestion/edgar_fetcher.py       # skips cached tickers
python src/parsing/section_extractor.py    # skips already-processed tickers
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py
python src/modeling/train.py
python src/modeling/evaluate.py
```

---

## Architecture

### Data flow
```
scrape_ipo_universe.py  →  data/raw/ipo_list_override.csv
ipo_list.py             →  data/processed/ipo_universe.csv
edgar_fetcher.py        →  data/raw/filings/{TICKER}/*.html
                           data/processed/filing_manifest.csv
price_fetcher.py        →  data/processed/returns.csv
section_extractor.py    →  data/processed/sections/{ticker}.json   (one per ticker, 424B4 preferred)
handcrafted.py          →  data/processed/handcrafted_features.csv
embeddings.py           →  data/cache/embeddings.npz
multiples.py            →  data/processed/multiples_features.csv
market_context.py       →  data/processed/market_context_features.csv
train.py                →  data/processed/models/{target}_{variant}_{model}.pkl
                           data/processed/model_results.json        ← OVERWRITTEN each run
evaluate.py             →  data/processed/plots/shap_*.png          ← OVERWRITTEN each run
                           data/processed/evaluation_report.md      ← OVERWRITTEN each run
                           results_tracker.md                       ← APPENDED each run
```

### Module responsibilities
| Module | Role |
|--------|------|
| `config/settings.py` | All paths, constants, keyword lists |
| `src/ingestion/scrape_ipo_universe.py` | Scrapes stockanalysis.com; enriches with yfinance sectors |
| `src/ingestion/edgar_fetcher.py` | CIK lookup (ticker → name fallback), 424B4 preference, backoff, archive page fallback |
| `src/parsing/section_extractor.py` | HTML → named section JSON (one file per ticker) |
| `src/features/handcrafted.py` | VADER sentiment (on summary+business only), keyword densities, readability |
| `src/features/embeddings.py` | Weighted-average section embeddings, cached to `.npz` |
| `src/features/multiples.py` | Financial features from filing HTML via BS4+regex; capped at 300KB + 200 tables to prevent hangs |
| `src/features/market_context.py` | VIX, S&P, sector ETF (all trailing/as-of); IPO volume via leakage-free 30/90-day lookback |
| `src/modeling/train.py` | 4 models × 3 variants × N targets; class-balanced; outputs balanced accuracy + prediction split |
| `src/modeling/evaluate.py` | SHAP plots per target; Markdown report; auto-appends run summary to `results_tracker.md` |

---

## Key Conventions
- Always import paths from `config/settings.py` — never hardcode
- Cache expensive ops in `data/cache/` (prices, embeddings, market data)
- Every module has `if __name__ == "__main__":` block
- Log warnings for failed extractions, never crash — NaN is fine
- `CV_FOLDS=5`; always report std alongside mean
- `train.py` skips variants with < 30 samples

---

## Critical Setup
`EDGAR_USER_AGENT` in `config/settings.py` must be set to a real name/email before running `edgar_fetcher.py`.

---

## Leakage Tests

Two standalone scripts in `src/modeling-test-leakage/` validate that model performance reflects genuine signal rather than data leakage.

### Permutation test
Trains the same models as `train.py` but with randomly shuffled labels. A clean model should collapse to AUC ≈ 0.50. If AUC stays elevated on shuffled labels, features contain post-IPO information.

```bash
python src/modeling-test-leakage/permutation_test.py                        # default: label_1m, 20 shuffles
python src/modeling-test-leakage/permutation_test.py --target label_6m --shuffles 50
```

**Clean result:** Real AUC >> null mean, p-value < 0.05 for all models.  
**Leakage signal:** Null AUC consistently >> 0.50.

### Temporal split test
Trains on the chronologically earlier half of IPOs, evaluates on the later half. Catches year-level confounding from `is_hot_ipo_year`/`ipo_year` and tests whether signal generalises across time.

```bash
python src/modeling-test-leakage/temporal_split_test.py                     # default: label_1m, 50/50 split
python src/modeling-test-leakage/temporal_split_test.py --target label_6m --split 0.6
```

**Clean result:** Test AUC 0.52–0.60, modest train→test drop.  
**Confounding signal:** M2/M3 test AUC drops sharply while M1 holds — year features driving M2/M3.

### Output
Results saved to `data/processed/leakage-test-results/` as timestamped JSON files (`permutation_{target}_{timestamp}.json`, `temporal_{target}_{timestamp}.json`). These are committed to git for tracking across runs.

---

## Design Notes

### Class imbalance
6m and 1y targets are heavily imbalanced (~30-34% positive rate). Naive accuracy baseline is 66-70%. All models use `class_weight="balanced"` (LR, RF) or `scale_pos_weight` (XGBoost). **Rely on ROC-AUC and balanced accuracy, not raw accuracy.**

### Leakage guards
- All market features use data on or before IPO date
- IPO volume feature (`ipos_prior_30d`, `ipos_prior_90d`) counts only IPOs strictly before the current date — no same-month future leakage
- `is_hot_ipo_year` encodes 2020/2021; use temporal train/test splits (not random) to guard against year-level confounding

### Financial features
`total_proceeds_m = 100` placeholder for most IPOs (stockanalysis.com doesn't publish proceeds). Known large IPOs have accurate values in `KNOWN_PROCEEDS` dict. Multiples extraction uses BeautifulSoup + regex on filing HTML; many fields will be NaN (graceful failure).

### EDGAR filing retrieval
The submissions API `recent` array holds ~1000 filings. For older IPOs, `edgar_fetcher.py` automatically fetches archive pages (`CIK{n}-submissions-001.json`). Section extractor outputs one `{ticker}.json` per ticker (424B4 preferred over S-1 since manifest is sorted that way).
