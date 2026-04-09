# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does
Research system testing whether language in IPO filings (SEC S-1 / 424B4) predicts post-IPO stock performance. Primary question: **does text signal add alpha over fundamentals?**

**Baseline model variants** (M-series):
- **M1** — text only (handcrafted NLP + `all-MiniLM-L6-v2` sentence-transformer embeddings)
- **M2** — structured features (financial multiples + market context: VIX, S&P, sector ETF, IPO volume)
- **M3** — M1 + M2 combined

**Enhanced experiment variants** (E-series, `enhanced_v2_no_ipoyear`):
- **E1** — M1 features + use-of-proceeds scores
- **E2** — M2 features + underwriter tier features + proceeds scores + year-relative normalized features
- **E3** — E1 + E2 combined

**PCA variants** (P-series, `pca_v1` / `pca_v1_tuned`): replace 384-dim embeddings with 30 PCA components (~70.7% variance). P1/P2/P3 use calendar-year regime normalization.

**PCA v2 variants** (P_v2-series, `pca_v2` / `pca_v2_tuned_regime_unaware`): same PCA compression but swap calendar-year normalization for **rolling 360-day normalization** (leakage-free). Also excludes `is_hot_ipo_year` (regime shortcut flag) in addition to `ipo_year`.

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

### Full baseline pipeline (run in this order)

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

# 6. Compute baseline features
python src/features/handcrafted.py
python src/features/multiples.py
python src/features/embeddings.py

# 7. Train all variants × models × windows (~10-15 min)
python src/modeling/train.py                       # trains all 4 targets
python src/modeling/train.py --target label_1m     # single target only
python src/modeling/train.py --target label_1m --notes "description of this run"

# 8. Evaluate: SHAP plots + report (auto-appends to results_tracker.md)
python src/modeling/evaluate.py
python src/modeling/evaluate.py --run run_20260407_225709   # load specific run
python src/modeling/evaluate.py --notes "post-leakage-fix evaluation"
```

### View results
```bash
cat data/processed/evaluation_report.md
streamlit run app/streamlit_app.py   # interactive explorer: IPO table, SHAP charts, model scores
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

### Enhanced experiment pipeline

```bash
# Step 1 — compute new feature files (run once after section_extractor)
python src/features/underwriter.py          # → data/processed/underwriter_features.csv
python src/features/proceeds.py             # → data/processed/proceeds_features.csv
                                            #   data/processed/proceeds_raw_text.csv (debug)
python src/features/regime_normalized.py    # → data/processed/regime_normalized_features.csv (calendar-year z-scores)
python src/features/regime_normalized.py --rolling   # → data/processed/regime_normalized_rolling_features.csv (360-day rolling, leakage-free)
python src/features/pca_embeddings.py       # → data/processed/pca_embeddings.csv (30 components)
python src/features/pca_embeddings.py --n-components 50   # optional: more components

# Step 2 — train enhanced variants (saves under experiments/{name}/, never overwrites baseline)
python src/modeling/train_experiment.py \
    --experiment enhanced_v2_no_ipoyear \
    --variants enhanced \
    --notes "removed ipo_year to eliminate temporal shortcut learning"

# All targets (omit --target):
python src/modeling/train_experiment.py --experiment enhanced_v2_no_ipoyear --variants enhanced

# --variants baseline  → trains M1/M2/M3 into the experiment dir for clean comparison
# --variants enhanced  → trains E1/E2/E3
# --variants pca       → trains P1/P2/P3 (calendar-year regime normalization)
# --variants pca_v2   → trains P1_v2/P2_v2/P3_v2 (rolling 360d normalization, regime-unaware)
# --variants all       → trains all four families

# Step 2b — tune hyperparameters (optional, run after initial training to find better params)
python src/modeling/tune_hyperparams.py \
    --experiment pca_v1 \
    --variants pca \
    --target label_1m \
    --notes "tuning tree depths and regularization"
# → saves data/processed/experiments/pca_v1/tuned_params.json
# Tune all 4 targets at once (omit --target):
python src/modeling/tune_hyperparams.py --experiment pca_v2 --variants pca_v2

# Step 2c — retrain with tuned params (new experiment dir, never overwrites the tuned source)
python src/modeling/train_experiment.py \
    --experiment pca_v1_tuned \
    --variants pca \
    --target label_1m \
    --hyperparams data/processed/experiments/pca_v1/tuned_params.json \
    --notes "pca_v1 with tuned hyperparams"
# pca_v2 equivalent (all 4 targets, per-target tuned params):
python src/modeling/train_experiment.py \
    --experiment pca_v2_tuned_regime_unaware \
    --variants pca_v2 \
    --hyperparams data/processed/experiments/pca_v2/tuned_params.json \
    --notes "pca_v2 with tuned hyperparams, rolling 360d regime-unaware normalization"

# Step 3 — compare baseline vs enhanced
python src/modeling/compare_experiments.py                         # auto-discovers latest runs
python src/modeling/compare_experiments.py \
    --baseline data/processed/run_results/run_XXXX.json \
    --enhanced data/processed/experiments/enhanced_v2_no_ipoyear/run_results/run_XXXX.json \
    --tag my_comparison

# Step 4 — bidirectional temporal generalization test
python src/modeling/temporal_bidirectional_test.py                              # enhanced variants, label_1m, 50/50 split
python src/modeling/temporal_bidirectional_test.py --variants pca               # test PCA v1 variants
python src/modeling/temporal_bidirectional_test.py --target label_6m
python src/modeling/temporal_bidirectional_test.py --target label_1m --split 0.6
# NOTE: pca_v2 is not yet a supported --variants choice in temporal_bidirectional_test.py
```

---

## Architecture

### Data flow
```
scrape_ipo_universe.py        →  data/raw/ipo_list_override.csv
ipo_list.py                   →  data/processed/ipo_universe.csv
edgar_fetcher.py              →  data/raw/filings/{TICKER}/*.html
                                 data/processed/filing_manifest.csv
price_fetcher.py              →  data/processed/returns.csv
section_extractor.py          →  data/processed/sections/{ticker}.json
handcrafted.py                →  data/processed/handcrafted_features.csv
embeddings.py                 →  data/cache/embeddings.npz
multiples.py                  →  data/processed/multiples_features.csv
market_context.py             →  data/processed/market_context_features.csv
underwriter.py                →  data/processed/underwriter_features.csv
proceeds.py                   →  data/processed/proceeds_features.csv
                                 data/processed/proceeds_raw_text.csv       (debug snippets)
regime_normalized.py          →  data/processed/regime_normalized_features.csv          (calendar-year mode, default)
regime_normalized.py --rolling→  data/processed/regime_normalized_rolling_features.csv   (rolling 360d mode)
pca_embeddings.py             →  data/processed/pca_embeddings.csv
                                 data/processed/pca_embeddings_meta.json

train.py                      →  data/processed/models/{target}_{variant}_{model}.pkl
                                 data/processed/model_results.json          ← OVERWRITTEN each run
                                 data/processed/run_results/{run_id}.json   ← NEVER overwritten
evaluate.py                   →  data/processed/plots/shap_*.png            ← OVERWRITTEN each run
                                 data/processed/evaluation_report.md        ← OVERWRITTEN each run
                                 results_tracker.md                         ← APPENDED each run

tune_hyperparams.py           →  data/processed/experiments/{name}/tuned_params.json
                                 data/processed/experiments/{name}/tuning_report.json
train_experiment.py           →  data/processed/experiments/{name}/models/
                                 data/processed/experiments/{name}/plots/shap_*.png
                                 data/processed/experiments/{name}/run_results/{run_id}.json
                                 data/processed/experiments/{name}/feature_manifest.json
compare_experiments.py        →  data/processed/comparisons/{tag}/comparison_long.csv
                                 data/processed/comparisons/{tag}/comparison_pivot.csv
                                 data/processed/comparisons/{tag}/comparison_config.json
temporal_bidirectional_test.py → results/temporal_bidirectional/results_{target}_{ts}.csv
                                 data/processed/experiments/temporal-bidirectional/plots/
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
| `src/features/market_context.py` | VIX, S&P, sector ETF (all trailing/as-of); IPO volume via leakage-free 30/90-day lookback; writes `ipo_year` (raw int — excluded from enhanced experiments) |
| `src/features/underwriter.py` | Extracts lead underwriter from section text + raw HTML fallback; normalizes via `NORMALIZATION_PATTERNS`; assigns tiers via `TIER_MAP` (both easy to edit at top of file) |
| `src/features/proceeds.py` | Keyword scoring of use-of-proceeds section into debt / growth / general / secondary; saves debug snippet CSV |
| `src/features/regime_normalized.py` | Two modes: default (calendar-year z-score/pctile, suffix `_year_z`/`_year_pctile`) and `--rolling` (360-day rolling window, suffix `_roll360_z`/`_roll360_pctile`, leakage-free). Groups by `ipo_year` in default mode but does not pass `ipo_year` itself as a model feature |
| `src/features/pca_embeddings.py` | Compresses 384-dim embeddings → N principal components (default 30, ~70.7% variance); StandardScaler before PCA; saves `pca_embeddings.csv` + `pca_embeddings_meta.json`; `--n-components` CLI arg |
| `src/modeling/train.py` | Baseline pipeline: 4 models × 3 variants (M1/M2/M3) × N targets; `--notes` flag; saves timestamped run JSON to `run_results/` |
| `src/modeling/evaluate.py` | SHAP plots per target; Markdown report; auto-appends run summary to `results_tracker.md`; `--run` flag to load specific run; `generate_shap_plot` accepts `output_dir` param |
| `src/modeling/train_experiment.py` | Enhanced pipeline: trains E1/E2/E3 (and optionally M1/M2/M3) under versioned experiment dir; generates SHAP plots post-training; `EXCLUDE_COLS` controls feature exclusions |
| `src/modeling/compare_experiments.py` | Loads two run JSONs (baseline + enhanced), produces long-form CSV + pivot table + delta AUC console summary |
| `src/modeling/temporal_bidirectional_test.py` | Strict chronological 50/50 split; trains in both directions (A→B and B→A); reports train AUC, test AUC, drop, balanced accuracy; `--variants` and `--hyperparams` flags |
| `src/modeling/tune_hyperparams.py` | RandomizedSearchCV (40 trials for trees, 8 for linear) per variant × model; saves `tuned_params.json` + `tuning_report.json`; prints default vs tuned AUC comparison table |

---

## Key Conventions
- Always import paths from `config/settings.py` — never hardcode
- Cache expensive ops in `data/cache/` (prices, embeddings, market data)
- Every module has `if __name__ == "__main__":` block
- Log warnings for failed extractions, never crash — NaN is fine
- `CV_FOLDS=5`; always report std alongside mean
- `train.py` and `train_experiment.py` skip variants with < 30 samples
- Use `--notes` on every training run — stored in run JSON for reproducibility
- Never name a new experiment the same as an existing one; use descriptive tags (e.g. `enhanced_v2_no_ipoyear`)

---

## Critical Setup
`EDGAR_USER_AGENT` in `config/settings.py` must be set to a real name/email before running `edgar_fetcher.py`.

---

## Experiment Design

### Baseline vs Enhanced
| Family | Variants | Script | Output dir |
|--------|----------|--------|------------|
| Baseline | M1_text, M2_multiples, M3_combined | `train.py` | `data/processed/models/` |
| Enhanced | E1_text_enhanced, E2_structured_enhanced, E3_combined_enhanced | `train_experiment.py --variants enhanced` | `data/processed/experiments/{name}/` |
| PCA v1 | P1_text_pca, P2_structured, P3_combined_pca | `train_experiment.py --variants pca` | `data/processed/experiments/{name}/` |
| PCA v2 | P1_v2_text_pca, P2_v2_structured, P3_v2_combined_pca | `train_experiment.py --variants pca_v2` | `data/processed/experiments/{name}/` |

P1/P1_v2 = handcrafted NLP + 30 PCA embedding components + proceeds  
P2 = same as E2 (structured, calendar-year normalization)  
P2_v2 = same as P2 but uses rolling 360d normalization (`regime_normalized_rolling_features.csv`)  
P3/P3_v2 = corresponding P1 + P2 combination

**Active experiments:** `enhanced_v2_no_ipoyear`, `pca_v1`, `pca_v1_tuned`, `pca_v2` (tuning only), `pca_v2_tuned_regime_unaware`

### EXCLUDE_COLS — feature exclusion list
Defined at the top of `train_experiment.py`. Currently: `{"ipo_year", "is_hot_ipo_year"}`.

- `ipo_year` — raw integer encodes calendar year, enabling temporal shortcut learning ("2021 = good"). Year-normalized features (`*_year_z`, `*_year_pctile`) use it as a grouping variable but are cross-sectionally normalized — **retained**.
- `is_hot_ipo_year` — binary flag for 2020/2021 directly encodes the hot IPO regime; excluded so models learn market conditions from continuous features only.

To add more exclusions, edit `EXCLUDE_COLS` in `train_experiment.py`. Do not modify `train.py` — baseline models intentionally have `ipo_year` available so results are comparable to prior literature.

### Run JSON envelope structure
Both `train.py` and `train_experiment.py` save a timestamped JSON per run:
```json
{
  "run_id":           "run_YYYYMMDD_HHMMSS",
  "timestamp":        "ISO datetime",
  "experiment":       "enhanced_v2_no_ipoyear",
  "notes":            "human-readable description",
  "targets_trained":  ["label_1m"],
  "hyperparameters":  { ... },
  "shap_plots":       { "label_1m/E1_text_enhanced/xgboost": "/path/to/plot.png" },
  "results":          { ... }
}
```
`evaluate.py` appends an `"evaluation"` block (timestamp, notes, shap_paths, report_path) to the run JSON after evaluation.

### Comparison workflow
`compare_experiments.py` matches variants by **variant_group**:
| Variant | Group |
|---------|-------|
| M1_text / E1_text_enhanced | text |
| M2_multiples / E2_structured_enhanced | structured |
| M3_combined / E3_combined_enhanced | combined |

Delta columns (`delta_auc_vs_baseline`, `delta_bal_acc_vs_baseline`) compare enhanced to baseline within the same (target, model, variant_group).

---

## Leakage Tests

### Permutation test (existing)
Trains the same models as `train.py` but with randomly shuffled labels. A clean model should collapse to AUC ≈ 0.50.

```bash
python src/modeling-test-leakage/permutation_test.py                        # default: label_1m, 20 shuffles
python src/modeling-test-leakage/permutation_test.py --target label_6m --shuffles 50
```

**Clean result:** Real AUC >> null mean, p-value < 0.05 for all models.
**Leakage signal:** Null AUC consistently >> 0.50.

### One-directional temporal split test (existing)
Trains on the chronologically earlier half, evaluates on the later half.

```bash
python src/modeling-test-leakage/temporal_split_test.py                     # default: label_1m, 50/50 split
python src/modeling-test-leakage/temporal_split_test.py --target label_6m --split 0.6
```

**Clean result:** Test AUC 0.52–0.60, modest train→test drop.
**Confounding signal:** M2/M3 test AUC drops sharply while M1 holds — year features driving M2/M3.

Results: `data/processed/leakage-test-results/` (timestamped JSON, committed to git).

### Bidirectional temporal generalization test (new)
Runs both directions (A→B and B→A) on enhanced experiment variants. More diagnostic than one-directional split.

```bash
python src/modeling/temporal_bidirectional_test.py --target label_1m
```

**Interpretation of label_1m results (enhanced_v2_no_ipoyear):**
- Split boundary: 2021-06-24 (258 earlier / 259 later IPOs)
- E1 text generalises best (avg test AUC 0.573 across both directions)
- E2 structured shows asymmetry: A→B 0.556 vs B→A 0.502 (structured features learned on early period don't transfer as well to post-2021)
- All models show large train/test AUC gaps (train ≈ 1.0 for E1/E3) — embeddings + tree models overfit on half-split training sets; prefer linear models for generalization

**Interpretation of label_1m results (pca_v1):**
- P1_text_pca avg test AUC: **0.603** (vs E1: 0.573) — PCA compression improves generalization +0.030
- Best run: P1 Ridge A→B test AUC **0.645**, drop only +0.161 (vs E1 Ridge drop ~0.26)
- RF/XGB still overfit even with 30 PCA dims (drops of 0.37–0.44) — **use LR/Ridge for temporal generalization**
- A→B consistently better than B→A across all variants (~+0.04–0.06 AUC) — post-2021 patterns generalize to pre-2021 better than the reverse
- `--variants pca` flag added to `temporal_bidirectional_test.py` for direct comparison

Results: `results/temporal_bidirectional/results_{target}_{timestamp}.csv`

---

## Design Notes

### Class imbalance
6m and 1y targets are heavily imbalanced (~30-34% positive rate). Naive accuracy baseline is 66-70%. All models use `class_weight="balanced"` (LR, RF) or `scale_pos_weight` (XGBoost). **Rely on ROC-AUC and balanced accuracy, not raw accuracy.**

### Leakage guards
- All market features use data on or before IPO date
- IPO volume (`ipos_prior_30d`, `ipos_prior_90d`) counts only IPOs strictly before the current date — no same-month future leakage
- `is_hot_ipo_year` encodes 2020/2021; use temporal train/test splits (not random) to guard against year-level confounding
- `ipo_year` (raw integer) is in `EXCLUDE_COLS` for all enhanced experiments — it was the primary temporal shortcut leakage vector
- `ipos_same_month` / `ipos_same_quarter` were old features removed due to leakage; replaced by `ipos_prior_30d` / `ipos_prior_90d`

### Financial features
`total_proceeds_m = 100` placeholder for most IPOs (stockanalysis.com doesn't publish proceeds). Known large IPOs have accurate values in `KNOWN_PROCEEDS` dict. Multiples extraction uses BeautifulSoup + regex on filing HTML; many fields will be NaN (graceful failure).

### Underwriter features
`NORMALIZATION_PATTERNS` and `TIER_MAP` are defined at the top of `underwriter.py` — edit those dicts to add aliases or reclassify banks. Tier 1 = major global IBs, Tier 2 = recognized mid-tier, Tier 3 = default for everything else. Current identification rate: ~81% of IPOs.

### EDGAR filing retrieval
The submissions API `recent` array holds ~1000 filings. For older IPOs, `edgar_fetcher.py` automatically fetches archive pages (`CIK{n}-submissions-001.json`). Section extractor outputs one `{ticker}.json` per ticker (424B4 preferred over S-1 since manifest is sorted that way).

---

## Research Findings (as of Apr 2026)

**Full run history:** `results_tracker.md` (auto-appended by `evaluate.py`).

Key findings so far:
- **M1 text consistently outperforms M2 structured** across runs — language signal in filings has more predictive power than financial multiples + market context alone
- **M3 adds marginal lift over M1** — structured features add ~0–3% AUC when text is already present
- **AUC range: 0.55–0.65** for `label_1m`; weak-signal territory — sample size (~425 IPOs) is the primary constraint
- **`ipo_year` was the primary leakage vector** in baseline M2/M3 — it encoded calendar year as an integer, enabling temporal shortcut learning; removed from all enhanced experiments
- **E1 text generalizes best temporally** (avg test AUC 0.573 in bidirectional split); E2 structured shows regime asymmetry (A→B 0.556 vs B→A 0.502)
- **Overfitting is the current bottleneck**: E1/E3 train AUC ≈ 1.0 in temporal splits — embeddings + deep trees memorize training half
- **PCA on embeddings (pca_v1, Apr 2026)**: 384→30 dims (70.7% variance) improved text temporal generalization E1 avg 0.573 → P1 avg **0.603**. Key insight: **linear models (LR/Ridge) generalize far better than trees** — tree drops 0.37–0.44 vs linear 0.16–0.27. The signal lives in linear combinations, not nonlinear interactions.
- **Hyperparameter tuning (pca_v1_tuned, Apr 2026)**: `tune_hyperparams.py` (RandomizedSearchCV, 40 trials) found shallower XGB (depth 2–3, subsample 0.6, min_child_weight 3) cuts XGB overfitting drop from +0.42 → **+0.30**. Ridge benefits from much stronger regularization (alpha 50–100 vs default 1.0). P1 Ridge A→B now hits test AUC **0.651**, drop +0.151 — project best for single holdout run. P3_combined_pca RF (tuned) hits CV AUC **0.676**.

- **pca_v1_tuned (all 4 targets, Apr 2026):** Per-target tuned params. Best results: P3 RF **0.744 AUC at label_1y**, P2 RF 0.700 at label_6m. Pattern: longer horizons have stronger signal (1w~0.60 → 1y~0.74). Text dominates at short horizons; structured features dominate at 6m; combined best everywhere.
- **pca_v2 tuning (Apr 2026):** Rolling 360d normalization + `is_hot_ipo_year` excluded. Tuning results (CV AUC from `tune_hyperparams.py`): P3_v2 XGB **0.767 at label_1y** (vs pca_v1_tuned P3 RF 0.744), P2_v2 XGB 0.715 at label_1y, P2_v2 RF 0.690 at label_6m, P3_v2 RF 0.695 at label_6m. P1_v2 text: 0.606–0.690 range across targets. Rolling normalization appears competitive with calendar-year at longer horizons; training of `pca_v2_tuned_regime_unaware` was interrupted before label_1y P3 completed — **resume command below in planned next steps**.

**Planned next steps:**
- **Resume `pca_v2_tuned_regime_unaware`** (interrupted — missing `label_1y` P2 XGB + all P3 models, no run JSON saved): `python src/modeling/train_experiment.py --experiment pca_v2_tuned_regime_unaware --variants pca_v2 --hyperparams data/processed/experiments/pca_v2/tuned_params.json --notes "pca_v2 with tuned hyperparams, rolling 360d regime-unaware normalization"`
- Run bidirectional temporal test on pca_v2 variants (requires adding `pca_v2` to `temporal_bidirectional_test.py` choices)
- Add regression target (predicted return %) alongside binary classification
- Investigate why B→A direction is consistently weaker (pre-2021 → post-2021 transfer is harder)
- Consider expanding training data (more filings) as the primary lever for further improvement

---

## Known Active Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| `sentence-transformers` not in `requirements.txt` | Medium | `pip install sentence-transformers` before running `embeddings.py` |
| Sector = "Unknown" for all 700 IPOs | Medium | Run `scrape_ipo_universe.py` without `--no-sector` to fix; affects sector ETF features in M2/M3 |
| `total_proceeds_m = 100` placeholder | Low | Most IPOs use placeholder; large IPOs have accurate values in `KNOWN_PROCEEDS` dict in `settings.py` |
| E1/E3 train AUC ≈ 1.0 in temporal split | Medium | Embeddings + tree models overfit on half-dataset training splits; use linear models (LR, Ridge) when evaluating temporal generalization |
| `proceeds_*_year_pctile` features have 0% coverage | Low | `proceeds_to_revenue_ratio` is nearly all NaN; silently produces all-NaN normalized columns — these get dropped by the all-NaN filter at training time |
