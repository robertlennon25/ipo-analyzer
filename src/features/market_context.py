"""
market_context.py
-----------------
Compute market regime features as of each IPO date.

Features (all computed from data available ON OR BEFORE the IPO date):
  - vix_on_ipo_date        : VIX close on IPO date (fear gauge)
  - vix_30d_avg            : VIX 30-day trailing average
  - sp500_ret_30d          : S&P 500 trailing 30-calendar-day return
  - sp500_ret_90d          : S&P 500 trailing 90-calendar-day return
  - sector_etf_ret_30d     : Sector ETF trailing 30-day return
  - ipos_same_month        : # of other IPOs in the same calendar month (hot market proxy)
  - ipos_same_quarter      : # of other IPOs in the same calendar quarter
  - is_hot_ipo_year        : 1 if year is in {2020, 2021}, else 0
  - market_regime          : categorical — bull / bear / neutral (based on sp500_ret_90d)

** Leakage note **
All features use trailing market data computed UP TO (not past) the IPO date.
There is no forward-looking bias in the strict sense.

However, two subtler risks are worth flagging:
  1. Year-level confounding: is_hot_ipo_year encodes 2020/2021 as special. If those
     years also produced genuinely better IPO returns (they did), the model may learn
     to predict "good" just because it's 2021 rather than because of text or fundamentals.
     Mitigation: always evaluate on held-out years (temporal split), not random splits.
  2. Market momentum autocorrelation: sp500_ret_30d is correlated with future short-term
     returns (momentum). The model may learn this rather than IPO-specific signal.
     Mitigation: run ablations with / without market context features and compare M2 vs M3.

Output: data/processed/market_context_features.csv
"""

import sys
import time
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CACHE_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UNIVERSE_PATH = PROCESSED_DIR / "ipo_universe.csv"
OUTPUT_PATH   = PROCESSED_DIR / "market_context_features.csv"
PRICE_CACHE   = CACHE_DIR / "market_price_cache.csv"

# Sector → ETF mapping (GICS-based)
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology":             "XLK",
    "Information Technology": "XLK",
    "Financials":             "XLF",
    "Financial Services":     "XLF",
    "Health Care":            "XLV",
    "Healthcare":             "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical":      "XLY",
    "Consumer Staples":       "XLP",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
    "Telecommunications":     "XLC",
}
DEFAULT_ETF = "^GSPC"  # fallback for Unknown sector (already in MARKET_TICKERS)

# Years considered "hot" IPO markets
HOT_IPO_YEARS = {2020, 2021}

# Tickers we'll need market data for
MARKET_TICKERS = ["^VIX", "^GSPC"] + sorted(set(SECTOR_ETF_MAP.values()))


# ---------------------------------------------------------------------------
# Market data loader (cached)
# ---------------------------------------------------------------------------

def load_market_data(start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV for all market tickers. Uses a parquet cache keyed by
    the date range — refreshes if the cached range doesn't cover what we need.
    """
    need_download = True

    if PRICE_CACHE.exists():
        try:
            cached = pd.read_csv(PRICE_CACHE, parse_dates=["date"])
            cached = cached.set_index(["ticker", "date"])
            cached_tickers = cached.index.get_level_values("ticker").unique().tolist()
            cached_start = str(cached.index.get_level_values("date").min().date())
            cached_end   = str(cached.index.get_level_values("date").max().date())
            covers_range   = (cached_start <= start) and (cached_end >= end)
            covers_tickers = all(t in cached_tickers for t in MARKET_TICKERS)
            if covers_range and covers_tickers:
                logger.info("Market data loaded from cache (%s → %s)", cached_start, cached_end)
                need_download = False
                raw = cached
        except Exception as exc:
            logger.warning("Cache read failed: %s — re-downloading", exc)

    if need_download:
        logger.info(
            "Downloading market data for %d tickers (%s → %s)...",
            len(MARKET_TICKERS), start, end,
        )
        frames = []
        for ticker in MARKET_TICKERS:
            try:
                hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
                if hist.empty:
                    logger.warning("No data for %s", ticker)
                    continue
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                hist.index.name = "date"
                hist["ticker"] = ticker
                frames.append(hist)
                time.sleep(0.3)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("Could not download any market data")

        raw = pd.concat(frames).reset_index().set_index(["ticker", "date"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw.reset_index().to_csv(PRICE_CACHE, index=False)
        logger.info("Market data cached: %s", PRICE_CACHE)

    # Split back into per-ticker DataFrames keyed by ticker symbol
    result: dict[str, pd.DataFrame] = {}
    for ticker in MARKET_TICKERS:
        try:
            result[ticker] = raw.loc[ticker].sort_index()
        except KeyError:
            pass
    return result


# ---------------------------------------------------------------------------
# Feature computation per IPO
# ---------------------------------------------------------------------------

def _trailing_return(price_series: pd.Series, as_of: pd.Timestamp, days: int) -> float | None:
    """
    Trailing calendar-day return ending on `as_of`.
    Uses Close prices. Returns None if insufficient history.
    """
    start = as_of - timedelta(days=days)
    window = price_series[(price_series.index >= start) & (price_series.index <= as_of)]
    if len(window) < 2:
        return None
    return float((window.iloc[-1] - window.iloc[0]) / window.iloc[0])


def _closest_value(series: pd.Series, as_of: pd.Timestamp) -> float | None:
    """Get the most recent value at or before `as_of`."""
    available = series[series.index <= as_of]
    if available.empty:
        return None
    return float(available.iloc[-1])


def compute_features_for_ipo(
    row: pd.Series,
    market_data: dict[str, pd.DataFrame],
    ipo_counts: pd.Series,
    ipo_quarter_counts: pd.Series,
) -> dict:
    """Compute all market context features for a single IPO."""
    ticker   = row["ticker"]
    ipo_date = pd.Timestamp(row["ipo_date"])
    sector   = str(row.get("sector", "Unknown"))
    year     = ipo_date.year
    month_key    = (year, ipo_date.month)
    quarter_key  = (year, ipo_date.quarter)

    features: dict = {"ticker": ticker}

    # --- VIX ---
    if "^VIX" in market_data:
        vix = market_data["^VIX"]["Close"]
        features["vix_on_ipo_date"] = _closest_value(vix, ipo_date)
        features["vix_30d_avg"] = (
            float(vix[(vix.index >= ipo_date - timedelta(days=30)) & (vix.index <= ipo_date)].mean())
            if not vix.empty else None
        )
    else:
        features["vix_on_ipo_date"] = None
        features["vix_30d_avg"] = None

    # --- S&P 500 ---
    if "^GSPC" in market_data:
        sp = market_data["^GSPC"]["Close"]
        features["sp500_ret_30d"] = _trailing_return(sp, ipo_date, 30)
        features["sp500_ret_90d"] = _trailing_return(sp, ipo_date, 90)
        ret_90 = features["sp500_ret_90d"]
        if ret_90 is None:
            features["market_regime"] = None
        elif ret_90 > 0.05:
            features["market_regime"] = "bull"
        elif ret_90 < -0.05:
            features["market_regime"] = "bear"
        else:
            features["market_regime"] = "neutral"
    else:
        features["sp500_ret_30d"] = None
        features["sp500_ret_90d"] = None
        features["market_regime"] = None

    # --- Sector ETF ---
    etf = SECTOR_ETF_MAP.get(sector, DEFAULT_ETF)
    features["sector_etf"] = etf
    if etf in market_data:
        etf_close = market_data[etf]["Close"]
        features["sector_etf_ret_30d"] = _trailing_return(etf_close, ipo_date, 30)
        features["sector_etf_ret_90d"] = _trailing_return(etf_close, ipo_date, 90)
    else:
        features["sector_etf_ret_30d"] = None
        features["sector_etf_ret_90d"] = None

    # Sector ETF outperformance vs S&P
    if features.get("sector_etf_ret_30d") is not None and features.get("sp500_ret_30d") is not None:
        features["sector_vs_sp500_30d"] = features["sector_etf_ret_30d"] - features["sp500_ret_30d"]
    else:
        features["sector_vs_sp500_30d"] = None

    # --- IPO volume (hot market proxy) ---
    features["ipos_same_month"]   = int(ipo_counts.get(month_key, 1)) - 1    # exclude self
    features["ipos_same_quarter"] = int(ipo_quarter_counts.get(quarter_key, 1)) - 1

    # --- Year regime ---
    features["is_hot_ipo_year"] = int(year in HOT_IPO_YEARS)
    features["ipo_year"] = year

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_market_context_features(
    universe_path: Path = UNIVERSE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Build market context features for all IPOs in the universe."""
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe not found: {universe_path}. Run ipo_list.py first.")

    universe = pd.read_csv(universe_path, parse_dates=["ipo_date"])
    print(f"Computing market context for {len(universe)} IPOs...")

    # Date range for market data: earliest IPO minus 120 days → latest IPO + 1 day
    earliest = (universe["ipo_date"].min() - timedelta(days=120)).strftime("%Y-%m-%d")
    latest   = (universe["ipo_date"].max() + timedelta(days=2)).strftime("%Y-%m-%d")

    market_data = load_market_data(start=earliest, end=latest)

    # Pre-compute IPO counts per month/quarter (used as hot-market proxy)
    universe["_month_key"]   = list(zip(universe["ipo_date"].dt.year, universe["ipo_date"].dt.month))
    universe["_quarter_key"] = list(zip(universe["ipo_date"].dt.year, universe["ipo_date"].dt.quarter))
    ipo_counts         = universe["_month_key"].value_counts()
    ipo_quarter_counts = universe["_quarter_key"].value_counts()

    records = []
    for _, row in universe.iterrows():
        try:
            feats = compute_features_for_ipo(row, market_data, ipo_counts, ipo_quarter_counts)
        except Exception as exc:
            logger.warning("Feature computation failed for %s: %s", row["ticker"], exc)
            feats = {"ticker": row["ticker"]}
        records.append(feats)

    df = pd.DataFrame(records)

    # One-hot encode market_regime (bull/bear/neutral) for model consumption
    if "market_regime" in df.columns:
        dummies = pd.get_dummies(df["market_regime"], prefix="regime")
        df = pd.concat([df, dummies], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print(f"\nSaved: {output_path} ({len(df)} rows, {len(df.columns)} cols)")
    print("\nCoverage per numeric field:")
    for col in numeric_cols:
        pct = f"{df[col].notna().mean():.0%}"
        print(f"  {col:35s} {pct}")

    print("\nSample (first 5 rows):")
    preview_cols = [
        "ticker", "vix_on_ipo_date", "sp500_ret_30d", "sp500_ret_90d",
        "sector_etf_ret_30d", "sector_vs_sp500_30d",
        "ipos_same_month", "is_hot_ipo_year", "market_regime",
    ]
    print(df[[c for c in preview_cols if c in df.columns]].head().to_string())

    return df


if __name__ == "__main__":
    df = build_market_context_features()
