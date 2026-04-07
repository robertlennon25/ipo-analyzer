"""
multiples.py
------------
Extract structured financial features from S-1 / 424B4 filings for M2/M3 model variants.

Extraction strategy:
- BeautifulSoup to parse HTML tables
- Regex to find dollar amounts near financial keywords
- Keyword-based signals for insider selling, price revisions, risk factor count
- Pull offer size and sector from ipo_universe.csv

Output: data/processed/multiples_features.csv
"""

import re
import sys
import logging
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, RAW_DIR  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FILINGS_DIR = RAW_DIR / "filings"
MANIFEST_PATH = PROCESSED_DIR / "filing_manifest.csv"
UNIVERSE_PATH = PROCESSED_DIR / "ipo_universe.csv"
OUTPUT_PATH = PROCESSED_DIR / "multiples_features.csv"

# Regex: captures dollar amounts in all common S-1 formats.
# Groups: (paren_val, paren_unit, dollar_val, dollar_unit, raw_val, raw_unit)
#   paren_*  → negative value in parentheses: (1,234) or ($200M)
#   dollar_* → explicit $ prefix: $1.2 billion, $500M
#   raw_*    → bare number with unit only: 200 million (no $ prefix)
AMOUNT_RE = re.compile(
    r"\(\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?\s*\)"
    r"|\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?"
    r"|([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)",
    re.IGNORECASE,
)

UNIT_MULTIPLIERS = {
    "billion": 1_000_000, "b": 1_000_000,
    "million": 1_000,     "m": 1_000,
    "thousand": 1,        "k": 1,
}

FINANCIAL_KEYWORDS: dict[str, list[str]] = {
    "revenue":      ["revenue", "net revenue", "total revenue", "revenues"],
    "gross_profit": ["gross profit"],
    "net_income":   ["net income", "net loss", "net income (loss)", "net loss (income)"],
    "ebitda":       ["ebitda", "adjusted ebitda"],
    "cash":         ["cash and cash equivalents", "cash, cash equivalents"],
    "total_assets": ["total assets"],
}

INSIDER_SELL_KEYWORDS = [
    "selling stockholder", "selling shareholder", "secondary shares",
    "existing stockholder", "insider selling",
]
PRICE_REVISION_KEYWORDS = [
    "amended", "revised", "increased the price range", "raised the price range",
    "increased our offering price",
]


# ---------------------------------------------------------------------------
# Dollar amount parsing
# ---------------------------------------------------------------------------

def _to_thousands(value_str: str, unit: str | None) -> float | None:
    """Convert a parsed number + unit to thousands of USD."""
    try:
        amount = float(value_str.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    multiplier = UNIT_MULTIPLIERS.get((unit or "").strip().lower(), 1)
    return amount * multiplier


def _parse_first_amount(text: str) -> float | None:
    """
    Extract the first dollar amount from a text snippet.
    Returns value in thousands USD, or None if nothing parseable found.
    """
    for m in AMOUNT_RE.finditer(text):
        paren_val, paren_unit, dollar_val, dollar_unit, raw_val, raw_unit = m.groups()
        if paren_val:
            val = _to_thousands(paren_val, paren_unit)
            return -val if val is not None else None
        if dollar_val:
            return _to_thousands(dollar_val, dollar_unit)
        if raw_val:
            return _to_thousands(raw_val, raw_unit)
    return None


# ---------------------------------------------------------------------------
# HTML table parsing
# ---------------------------------------------------------------------------

def _row_text(row_tag) -> str:
    return " ".join(row_tag.get_text(" ", strip=True).split())


MAX_TABLES = 200        # stop after this many tables — financial tables are near the front
MAX_TEXT_CHARS = 300_000  # truncate raw HTML before regex signal search (~first 300KB)


def _extract_from_tables(soup: BeautifulSoup) -> dict[str, list[float | None]]:
    """
    Scan all <table> tags and extract financial values by keyword matching.
    Returns up to 2 values per field (current year, prior year).
    """
    results: dict[str, list[float | None]] = {k: [] for k in FINANCIAL_KEYWORDS}

    for table in soup.find_all("table", limit=MAX_TABLES):
        for row in table.find_all("tr"):
            row_lower = _row_text(row).lower()

            for field, keywords in FINANCIAL_KEYWORDS.items():
                if len(results[field]) >= 2:
                    continue
                if not any(kw in row_lower for kw in keywords):
                    continue

                amounts: list[float | None] = []
                for cell in row.find_all(["td", "th"]):
                    cell_text = cell.get_text(" ", strip=True)
                    if len(cell_text) > 40:  # skip label cells
                        continue
                    val = _parse_first_amount(cell_text)
                    if val is not None:
                        amounts.append(val)

                if amounts and not results[field]:
                    results[field] = amounts[:2]
                    logger.debug(
                        "  %s: %s (row: '%s')", field, amounts[:2], row_lower[:60]
                    )

    return results


# ---------------------------------------------------------------------------
# Text-level signal extraction
# ---------------------------------------------------------------------------

def _find_insider_pct(text_lower: str) -> float | None:
    """Heuristic: find % of proceeds going to selling stockholders."""
    patterns = [
        r"selling stockholders will receive.*?(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%.*?proceeds.*?selling stockholder",
        r"(\d+(?:\.\d+)?)\s*%.*?secondary",
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


def _extract_text_signals(html_text: str) -> dict[str, float | int | None]:
    """Extract keyword-based signals from the full filing text."""
    signals: dict[str, float | int | None] = {}
    text_lower = html_text.lower()

    signals["has_insider_selling"] = int(
        any(kw in text_lower for kw in INSIDER_SELL_KEYWORDS)
    )
    signals["price_range_revised_up"] = int(
        any(kw in text_lower for kw in PRICE_REVISION_KEYWORDS)
    )

    risk_match = re.search(
        r"risk\s+factors(.*?)"
        r"(?:use\s+of\s+proceeds|dividend|dilution|capitalization|business)",
        html_text,
        re.IGNORECASE | re.DOTALL,
    )
    if risk_match:
        numbered = re.findall(r"(?:^|\n)\s*\d{1,2}[\.\)]\s+[A-Z]", risk_match.group(1))
        signals["risk_factor_count"] = len(numbered)
    else:
        signals["risk_factor_count"] = None

    signals["insider_proceeds_pct"] = _find_insider_pct(text_lower)
    return signals


# ---------------------------------------------------------------------------
# Feature derivation (split into two helpers to stay under branch/var limits)
# ---------------------------------------------------------------------------

def _raw_financials_to_dict(raw: dict) -> dict[str, float | None]:
    """Unpack raw extracted lists into a flat dict of base financial values."""

    def first(key: str) -> float | None:
        vals = raw.get(key, [])
        return vals[0] if vals else None

    def second(key: str) -> float | None:
        vals = raw.get(key, [])
        return vals[1] if len(vals) > 1 else None

    return {
        "revenue_current": first("revenue"),
        "revenue_prior":   second("revenue"),
        "gross_profit":    first("gross_profit"),
        "net_income":      first("net_income"),
        "ebitda":          first("ebitda"),
        "cash":            first("cash"),
        "total_assets":    first("total_assets"),
    }


def _derive_features(raw: dict, universe_row: pd.Series | None) -> dict[str, float | int | None]:
    """Compute derived financial features from raw extracted values."""
    base = _raw_financials_to_dict(raw)
    features: dict[str, float | int | None] = dict(base)

    rev = base["revenue_current"]
    rev_prior = base["revenue_prior"]
    gp = base["gross_profit"]
    ni = base["net_income"]
    cash = base["cash"]

    features["revenue_growth_pct"] = (
        (rev - rev_prior) / abs(rev_prior) * 100
        if rev is not None and rev_prior
        else None
    )
    features["gross_margin_pct"] = (
        gp / rev * 100 if gp is not None and rev else None
    )
    features["is_profitable"] = int(ni > 0) if ni is not None else None
    features["net_income_pct_revenue"] = (
        ni / rev * 100 if ni is not None and rev else None
    )
    features["cash_burn_proxy"] = (
        ni / cash if ni is not None and cash else None
    )

    if universe_row is not None:
        proceeds = universe_row.get("total_proceeds_m")
        features["total_proceeds_m"] = float(proceeds) if pd.notna(proceeds) else None
        features["sector"] = universe_row.get("sector")
        rev_m = (rev / 1000) if rev and rev > 0 else None
        features["proceeds_to_revenue_ratio"] = (
            float(proceeds) / rev_m if proceeds is not None and rev_m else None
        )
    else:
        features["total_proceeds_m"] = None
        features["sector"] = None
        features["proceeds_to_revenue_ratio"] = None

    return features


# ---------------------------------------------------------------------------
# Per-filing extraction
# ---------------------------------------------------------------------------

def extract_multiples_for_filing(
    file_path: Path, universe_row: pd.Series | None
) -> dict:
    """
    Parse one filing HTML file and return a dict of financial features.
    Fails gracefully — missing fields become NaN.
    """
    try:
        html_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return {}

    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception as e:  # bs4 can raise various internal errors
        logger.warning("Could not parse HTML %s: %s", file_path, e)
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            raw_financials = _extract_from_tables(soup)
        except Exception as e:
            logger.warning("Table extraction failed for %s: %s", file_path, e)
            raw_financials = {k: [] for k in FINANCIAL_KEYWORDS}

    try:
        text_signals = _extract_text_signals(html_text[:MAX_TEXT_CHARS])
    except Exception as e:
        logger.warning("Text signal extraction failed for %s: %s", file_path, e)
        text_signals = {}

    try:
        derived = _derive_features(raw_financials, universe_row)
    except Exception as e:
        logger.warning("Feature derivation failed for %s: %s", file_path, e)
        derived = {}

    return {**derived, **text_signals}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_multiples_features(
    manifest_path: Path = MANIFEST_PATH,
    universe_path: Path = UNIVERSE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Process all filings in the manifest and produce multiples_features.csv.
    Prefers 424B4 over S-1 when both are available for a ticker.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run edgar_fetcher.py first."
        )
    manifest = pd.read_csv(manifest_path)

    ok = manifest[manifest["status"].isin(["ok", "cached"])].copy()
    ok = ok[ok["file_path"].notna()]
    ok["file_path"] = ok["file_path"].apply(Path)
    ok = ok[ok["file_path"].apply(lambda p: p.exists())]

    # Prefer 424B4 (priority 0) over S-1 (priority 1); take most-recent if tied
    type_priority = {"424B4": 0, "S-1": 1}
    ok["_priority"] = ok["filing_type"].map(type_priority).fillna(2)
    ok = (
        ok.sort_values(["ticker", "_priority", "filing_date"], ascending=[True, True, False])
        .drop_duplicates(subset=["ticker"], keep="first")
    )

    print(f"Processing {len(ok)} filings (1 per ticker, preferring 424B4)...")

    universe: pd.DataFrame | None = None
    if universe_path.exists():
        universe = pd.read_csv(universe_path).set_index("ticker")
    else:
        logger.warning(
            "IPO universe not found at %s — sector/proceeds will be NaN", universe_path
        )

    records = []
    for _, row in ok.iterrows():
        ticker = row["ticker"]
        filing_type = row.get("filing_type", "unknown")
        file_path: Path = row["file_path"]
        print(f"  {ticker} ({filing_type}): {file_path.name}")

        univ_row = (
            universe.loc[ticker]
            if universe is not None and ticker in universe.index
            else None
        )
        features = extract_multiples_for_filing(file_path, univ_row)
        features["ticker"] = ticker
        features["filing_type"] = filing_type
        records.append(features)

    result = pd.DataFrame(records)
    front = ["ticker", "filing_type"]
    result = result[front + [c for c in result.columns if c not in front]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    print(f"\nSaved: {output_path} ({len(result)} rows, {len(result.columns)} cols)")
    print("Coverage per field:")
    for col in numeric_cols:
        pct = f"{result[col].notna().mean():.0%}"
        print(f"  {col:35s} {pct}")

    return result


if __name__ == "__main__":
    out = build_multiples_features()
    print("\nSample (rows with revenue data):")
    has_rev = out.dropna(subset=["revenue_current"])
    if len(has_rev):
        cols = [
            "ticker", "revenue_current", "revenue_growth_pct",
            "gross_margin_pct", "is_profitable", "net_income_pct_revenue",
            "total_proceeds_m", "sector",
        ]
        print(has_rev[cols].to_string())
    else:
        print("No revenue data extracted — check that filings are downloaded.")
