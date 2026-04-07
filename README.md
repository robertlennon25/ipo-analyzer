# IPO Language & Aftermarket Performance Analyzer

Research system that tests whether language in IPO filings (SEC S-1 / 424B4) predicts post-IPO stock performance. Primary question: **does text signal add alpha over fundamentals?**

## Results

See [`results_tracker.md`](results_tracker.md) for full run history. Summary of progression:

**Run 001 (Apr 2026)** — 425 samples, label_1m only, 2 model types, raw accuracy metric
| Variant | Best ROC-AUC |
|---------|-------------|
| M1 text | 0.612 ± 0.029 |
| M2 fundamentals | 0.535 ± 0.075 |
| M3 combined | 0.615 ± 0.038 |

**Run 002 (Apr 2026)** — more data, 4 model types, 4 return windows, class-balanced training, leakage fix on IPO volume feature — *see results_tracker.md for full breakdown*

**Key finding so far:** Text features (M1) consistently outperform structured financials (M2). M3 adds marginal lift over M1. Results are in weak-signal territory (0.55–0.65 AUC); sample size is the primary constraint.

---

## Model Variants

| Variant | Features | Research Question |
|---------|----------|-------------------|
| M1 | Handcrafted NLP + `all-MiniLM-L6-v2` embeddings | Pure language signal |
| M2 | Revenue/margins/proceeds + VIX/S&P/sector ETF/IPO volume | Pure fundamentals + market signal |
| M3 | M1 + M2 | Does text add alpha over numbers? |

Each variant trained with: **Logistic Regression**, **Ridge**, **Random Forest**, **XGBoost**  
Each model evaluated across four return windows: **1w, 1m, 6m, 1y**  
Primary metric: **ROC-AUC** (balanced accuracy also reported; raw accuracy is misleading at 6m/1y due to class imbalance)

---

## Setup

```bash
git clone https://github.com/robertlennon25/ipo-analyzer
cd ipo-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers
```

Set your EDGAR user agent in `config/settings.py`:
```python
EDGAR_USER_AGENT = "Your Name your@email.com"
```

## Running the Pipeline

```bash
# 1. Build IPO universe with sectors (~4 min)
python src/ingestion/scrape_ipo_universe.py
python src/ingestion/ipo_list.py

# 2. Fetch price returns (~5 min, cached)
python src/ingestion/price_fetcher.py

# 3. Download filings from EDGAR
python src/ingestion/edgar_fetcher.py --limit 50   # test; drop --limit for full run (3-5 hrs)

# 4. Market context (can run alongside step 3)
python src/features/market_context.py

# 5. Extract sections + features
python src/parsing/section_extractor.py
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py

# 6. Train and evaluate (results auto-appended to results_tracker.md)
python src/modeling/train.py
python src/modeling/evaluate.py
```

---

## Leakage Validation

Two tests in `src/modeling-test-leakage/` verify that model performance reflects genuine signal.

### Permutation test — shuffles labels to build a null distribution
```bash
python src/modeling-test-leakage/permutation_test.py
python src/modeling-test-leakage/permutation_test.py --target label_6m --shuffles 50
```
If AUC stays elevated on shuffled labels → features contain post-IPO information. Clean models should collapse to AUC ≈ 0.50.

### Temporal split test — trains on early IPOs, tests on later ones
```bash
python src/modeling-test-leakage/temporal_split_test.py
python src/modeling-test-leakage/temporal_split_test.py --target label_1m --split 0.6
```
Catches year-level confounding (`is_hot_ipo_year`) and tests whether signal generalises across time. If M2/M3 drops sharply while M1 holds, year features are driving structured model performance.

Results saved to `data/processed/leakage-test-results/` as timestamped JSON and committed to git.

---

## Data Sources

| Source | What | How |
|--------|------|-----|
| [stockanalysis.com](https://stockanalysis.com/ipos/) | IPO list 2019–2023 | Scraped |
| SEC EDGAR | S-1 / 424B4 filing HTML | EDGAR submissions API |
| yfinance | Post-IPO prices, sectors | REST |
| yfinance | VIX, S&P 500, sector ETFs | REST, cached |

---

## Notes on Data Quality

- `total_proceeds_m = 100` placeholder for most IPOs — stockanalysis.com doesn't publish proceeds in list view
- ~185/700 IPOs have unknown sector (delisted tickers not found by yfinance)
- 6m/1y targets are class-imbalanced (~30-34% positive rate) — use ROC-AUC, not accuracy
- Financial feature extraction (multiples.py) has high NaN rates for many filings
