"""
regime_normalized.py
--------------------
Compute year-relative normalized versions of regime-sensitive numeric features.

For each target column, produces:
  {col}_year_z       within-year z-score  (mean=0, std=1 per ipo_year group)
  {col}_year_pctile  within-year percentile rank  (0–1)

Groups with < MIN_GROUP_SIZE valid observations → NaN (avoids overfitting small years).

Features normalized from market_context_features.csv (grouped by ipo_year):
  vix_on_ipo_date, vix_30d_avg,
  sp500_ret_30d, sp500_ret_90d,
  ipos_prior_30d, ipos_prior_90d,
  sector_etf_ret_30d, sector_etf_ret_90d, sector_vs_sp500_30d

Features normalized from multiples_features.csv (joined on ticker → ipo_year):
  revenue_current, revenue_prior, revenue_growth_pct,
  gross_margin_pct, total_proceeds_m, proceeds_to_revenue_ratio

Why certain columns are skipped:
  is_hot_ipo_year  — already a year binary; within-year z-score is 0 by definition
  ipo_year         — the grouping variable itself
  regime_* (one-hots), market_regime, sector_etf  — categorical / string
  is_profitable, has_* binary flags — binary; z-score is misleading

Output: data/processed/regime_normalized_features.csv
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MARKET_PATH   = PROCESSED_DIR / "market_context_features.csv"
MULTIPLES_PATH = PROCESSED_DIR / "multiples_features.csv"
OUTPUT_PATH   = PROCESSED_DIR / "regime_normalized_features.csv"

MIN_GROUP_SIZE = 5  # minimum non-NaN observations per year to compute z-score

MARKET_COLS_TO_NORMALIZE: list[str] = [
    "vix_on_ipo_date",
    "vix_30d_avg",
    "sp500_ret_30d",
    "sp500_ret_90d",
    "ipos_prior_30d",
    "ipos_prior_90d",
    "sector_etf_ret_30d",
    "sector_etf_ret_90d",
    "sector_vs_sp500_30d",
]

MULTIPLES_COLS_TO_NORMALIZE: list[str] = [
    "revenue_current",
    "revenue_prior",
    "revenue_growth_pct",
    "gross_margin_pct",
    "total_proceeds_m",
    "proceeds_to_revenue_ratio",
]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _within_year_zscore(series: pd.Series, year_series: pd.Series) -> pd.Series:
    """Within-year z-score. Groups smaller than MIN_GROUP_SIZE → NaN."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for year, idx in series.groupby(year_series).groups.items():
        vals = series.loc[idx]
        n_valid = vals.notna().sum()
        if n_valid < MIN_GROUP_SIZE:
            continue
        mu = vals.mean()
        sigma = vals.std()
        if sigma == 0 or np.isnan(sigma):
            result.loc[idx] = 0.0  # constant within year → z = 0
        else:
            result.loc[idx] = (vals - mu) / sigma
    return result


def _within_year_pctile(series: pd.Series, year_series: pd.Series) -> pd.Series:
    """Within-year percentile rank (0–1). NaN observations remain NaN."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for year, idx in series.groupby(year_series).groups.items():
        vals = series.loc[idx]
        n_valid = vals.notna().sum()
        if n_valid < MIN_GROUP_SIZE:
            continue
        result.loc[idx] = vals.rank(pct=True, na_option="keep")
    return result


def _normalize_cols(
    df: pd.DataFrame,
    cols: list[str],
    year_col: pd.Series,
    output_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute z-score and pctile for each col and append to output_df."""
    for col in cols:
        if col not in df.columns:
            logger.warning("Column not found, skipping normalization: %s", col)
            continue
        n_valid = df[col].notna().sum()
        if n_valid < MIN_GROUP_SIZE:
            logger.info("Skipping %s — only %d non-NaN values (< %d threshold)", col, n_valid, MIN_GROUP_SIZE)
            continue
        output_df[f"{col}_year_z"]      = _within_year_zscore(df[col], year_col)
        output_df[f"{col}_year_pctile"] = _within_year_pctile(df[col], year_col)
    return output_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_regime_normalized_features(
    market_path: Path = MARKET_PATH,
    multiples_path: Path = MULTIPLES_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Load market context + multiples, compute normalized features, save CSV."""
    if not market_path.exists():
        raise FileNotFoundError(f"Market features not found: {market_path}. Run market_context.py first.")

    market_df = pd.read_csv(market_path)
    if "ipo_year" not in market_df.columns:
        raise ValueError("market_context_features.csv is missing ipo_year column.")

    output_df = market_df[["ticker", "ipo_year"]].copy()
    year_col  = market_df["ipo_year"]

    # Market context normalization
    output_df = _normalize_cols(market_df, MARKET_COLS_TO_NORMALIZE, year_col, output_df)

    # Multiples normalization (joined via ipo_year from market)
    if multiples_path.exists():
        mult_df = pd.read_csv(multiples_path)
        mult_with_year = mult_df.merge(market_df[["ticker", "ipo_year"]], on="ticker", how="left")
        year_col_mult = mult_with_year["ipo_year"]

        mult_norm = mult_with_year[["ticker"]].copy()
        mult_norm = _normalize_cols(mult_with_year, MULTIPLES_COLS_TO_NORMALIZE, year_col_mult, mult_norm)

        output_df = output_df.merge(mult_norm, on="ticker", how="left")
    else:
        logger.warning("Multiples features not found — skipping multiples normalization.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    norm_cols = [c for c in output_df.columns if c.endswith("_year_z") or c.endswith("_year_pctile")]
    print(f"\nSaved: {output_path}  ({len(output_df)} rows, {len(norm_cols)} normalized columns)")
    print(f"\n{'Column':<45} {'Coverage':>8}")
    print("-" * 55)
    for col in norm_cols:
        pct = output_df[col].notna().mean()
        print(f"  {col:<43} {pct:>7.0%}")

    return output_df


if __name__ == "__main__":
    build_regime_normalized_features()
