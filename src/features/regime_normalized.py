"""
regime_normalized.py
--------------------
Compute regime-relative normalized versions of market and financial features.

Two modes:
  calendar-year (default, legacy):
    Groups by ipo_year → {col}_year_z / {col}_year_pctile
    Output: data/processed/regime_normalized_features.csv

  rolling (--rolling, leakage-free):
    For each IPO at date D, normalizes against all IPOs with dates in
    [D - window_days, D)  (strictly prior — no same-day, no future data).
    Output: data/processed/regime_normalized_rolling_features.csv
    Column suffix: {col}_roll360_z / {col}_roll360_pctile

Why rolling is preferred:
  Calendar-year normalization leaks forward: an IPO in January gets a
  z-score computed using December IPOs that haven't happened yet.
  Rolling window uses only past observations → strictly causal.

Usage:
  python src/features/regime_normalized.py                 # calendar-year (legacy)
  python src/features/regime_normalized.py --rolling       # 360-day rolling window
  python src/features/regime_normalized.py --rolling --window-days 180
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MARKET_PATH    = PROCESSED_DIR / "market_context_features.csv"
MULTIPLES_PATH = PROCESSED_DIR / "multiples_features.csv"
RETURNS_PATH   = PROCESSED_DIR / "returns.csv"
OUTPUT_YEAR    = PROCESSED_DIR / "regime_normalized_features.csv"
OUTPUT_ROLLING = PROCESSED_DIR / "regime_normalized_rolling_features.csv"

MIN_GROUP_SIZE = 5    # minimum observations in window/year before z-score is computed
DEFAULT_WINDOW = 360  # rolling window in days

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
# Calendar-year normalization (legacy)
# ---------------------------------------------------------------------------

def _within_year_zscore(series: pd.Series, year_series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for _, idx in series.groupby(year_series).groups.items():
        vals = series.loc[idx]
        if vals.notna().sum() < MIN_GROUP_SIZE:
            continue
        mu, sigma = vals.mean(), vals.std()
        result.loc[idx] = 0.0 if (sigma == 0 or np.isnan(sigma)) else (vals - mu) / sigma
    return result


def _within_year_pctile(series: pd.Series, year_series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for _, idx in series.groupby(year_series).groups.items():
        vals = series.loc[idx]
        if vals.notna().sum() < MIN_GROUP_SIZE:
            continue
        result.loc[idx] = vals.rank(pct=True, na_option="keep")
    return result


def _normalize_year(df: pd.DataFrame, cols: list[str],
                    year_col: pd.Series, out: pd.DataFrame) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            logger.warning("Column not found, skipping: %s", col)
            continue
        if df[col].notna().sum() < MIN_GROUP_SIZE:
            continue
        out[f"{col}_year_z"]      = _within_year_zscore(df[col], year_col)
        out[f"{col}_year_pctile"] = _within_year_pctile(df[col], year_col)
    return out


# ---------------------------------------------------------------------------
# Rolling-window normalization (leakage-free)
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, dates: pd.Series,
                    window_days: int = DEFAULT_WINDOW) -> pd.Series:
    """
    For each IPO at date D with value x, compute z-score against all IPOs
    with dates in [D - window_days, D).  Strictly prior — no leakage.
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    delta  = pd.Timedelta(days=window_days)

    for idx in series.index:
        x = series.loc[idx]
        if pd.isna(x):
            continue
        d = dates.loc[idx]
        mask = (dates >= d - delta) & (dates < d)
        window = series.loc[mask].dropna()
        if len(window) < MIN_GROUP_SIZE:
            continue
        mu, sigma = window.mean(), window.std()
        result.loc[idx] = 0.0 if (sigma == 0 or np.isnan(sigma)) else (x - mu) / sigma

    return result


def _rolling_pctile(series: pd.Series, dates: pd.Series,
                    window_days: int = DEFAULT_WINDOW) -> pd.Series:
    """
    For each IPO at date D with value x, compute what fraction of the
    [D - window_days, D) window values are ≤ x.  Strictly prior — no leakage.
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    delta  = pd.Timedelta(days=window_days)

    for idx in series.index:
        x = series.loc[idx]
        if pd.isna(x):
            continue
        d = dates.loc[idx]
        mask = (dates >= d - delta) & (dates < d)
        window = series.loc[mask].dropna()
        if len(window) < MIN_GROUP_SIZE:
            continue
        result.loc[idx] = float((window <= x).mean())

    return result


def _normalize_rolling(df: pd.DataFrame, cols: list[str],
                       dates: pd.Series, window_days: int,
                       out: pd.DataFrame) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            logger.warning("Column not found, skipping: %s", col)
            continue
        if df[col].notna().sum() < MIN_GROUP_SIZE:
            continue
        out[f"{col}_roll{window_days}_z"]      = _rolling_zscore(df[col], dates, window_days)
        out[f"{col}_roll{window_days}_pctile"] = _rolling_pctile(df[col], dates, window_days)
    return out


# ---------------------------------------------------------------------------
# Main pipelines
# ---------------------------------------------------------------------------

def build_calendar_year(
    market_path: Path = MARKET_PATH,
    multiples_path: Path = MULTIPLES_PATH,
    output_path: Path = OUTPUT_YEAR,
) -> pd.DataFrame:
    """Original calendar-year normalization (kept for backward compatibility)."""
    market_df = pd.read_csv(market_path)
    if "ipo_year" not in market_df.columns:
        raise ValueError("market_context_features.csv missing ipo_year column.")

    out = market_df[["ticker", "ipo_year"]].copy()
    out = _normalize_year(market_df, MARKET_COLS_TO_NORMALIZE, market_df["ipo_year"], out)

    if multiples_path.exists():
        mult = pd.read_csv(multiples_path).merge(
            market_df[["ticker", "ipo_year"]], on="ticker", how="left"
        )
        mult_out = mult[["ticker"]].copy()
        mult_out = _normalize_year(mult, MULTIPLES_COLS_TO_NORMALIZE, mult["ipo_year"], mult_out)
        out = out.merge(mult_out, on="ticker", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    _print_coverage(out, output_path)
    return out


def build_rolling(
    market_path: Path = MARKET_PATH,
    multiples_path: Path = MULTIPLES_PATH,
    returns_path: Path = RETURNS_PATH,
    output_path: Path = OUTPUT_ROLLING,
    window_days: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Rolling-window normalization.  Each IPO is normalized against the
    `window_days` days prior to its IPO date — strictly causal, no leakage.
    """
    market_df = pd.read_csv(market_path)

    if not returns_path.exists():
        raise FileNotFoundError(f"returns.csv not found at {returns_path}. Run price_fetcher.py first.")
    returns_df = pd.read_csv(returns_path, parse_dates=["ipo_date"])

    # Join ipo_date onto market features
    df = market_df.merge(returns_df[["ticker", "ipo_date"]], on="ticker", how="inner")
    df = df.sort_values("ipo_date").reset_index(drop=True)

    print(f"Rolling normalization: {len(df)} IPOs, window={window_days} days")
    dates = df["ipo_date"]

    out = df[["ticker"]].copy()
    out = _normalize_rolling(df, MARKET_COLS_TO_NORMALIZE, dates, window_days, out)

    if multiples_path.exists():
        mult = pd.read_csv(multiples_path).merge(df[["ticker", "ipo_date"]], on="ticker", how="inner")
        mult = mult.sort_values("ipo_date").reset_index(drop=True)
        mult_out = mult[["ticker"]].copy()
        mult_out = _normalize_rolling(mult, MULTIPLES_COLS_TO_NORMALIZE,
                                      mult["ipo_date"], window_days, mult_out)
        out = out.merge(mult_out, on="ticker", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    _print_coverage(out, output_path)
    return out


def _print_coverage(df: pd.DataFrame, path: Path) -> None:
    norm_cols = [c for c in df.columns if c.endswith("_z") or c.endswith("_pctile")]
    print(f"\nSaved: {path}  ({len(df)} rows, {len(norm_cols)} normalized columns)")
    print(f"\n{'Column':<50} {'Coverage':>8}")
    print("-" * 60)
    for col in norm_cols:
        pct = df[col].notna().mean()
        print(f"  {col:<48} {pct:>7.0%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute regime-normalized features")
    parser.add_argument("--rolling", action="store_true",
                        help="Use rolling window instead of calendar-year grouping (leakage-free).")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW,
                        help=f"Rolling window size in days (default: {DEFAULT_WINDOW}). "
                             "Only used with --rolling.")
    args = parser.parse_args()

    if args.rolling:
        build_rolling(window_days=args.window_days)
    else:
        build_calendar_year()
