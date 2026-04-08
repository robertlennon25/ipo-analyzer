"""
proceeds.py
-----------
Rule-based extraction and classification of Use of Proceeds content.

Four categories scored via keyword counting:
  1. debt       — refinancing / repaying loans / credit facilities
  2. growth     — capex, R&D, acquisitions, expansion, hiring
  3. general    — general corporate purposes, working capital
  4. secondary  — selling stockholder / secondary offering proceeds

Features per IPO (all numeric, model-ready):

  Raw keyword scores (count of distinct matching keywords):
    proceeds_debt_score, proceeds_growth_score,
    proceeds_general_score, proceeds_secondary_score

  Proportional features (each score ÷ total; NaN when no keywords hit):
    proceeds_debt_pct, proceeds_growth_pct,
    proceeds_general_pct, proceeds_secondary_pct

  Binary flags:
    has_debt_repayment_flag, has_growth_flag

  Metadata:
    proceeds_section_found  — 1 if use_of_proceeds section was found in JSON
    proceeds_text_length    — character count of extracted proceeds text

Debug file (for inspection): data/processed/proceeds_raw_text.csv
Output:                      data/processed/proceeds_features.csv
"""

import re
import sys
import json
import logging
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SECTIONS_DIR = PROCESSED_DIR / "sections"
OUTPUT_PATH  = PROCESSED_DIR / "proceeds_features.csv"
DEBUG_PATH   = PROCESSED_DIR / "proceeds_raw_text.csv"

# ---------------------------------------------------------------------------
# Keyword lists — edit freely to tune classification sensitivity.
# Each list contains substrings; a match = substring present (case-insensitive).
# ---------------------------------------------------------------------------

DEBT_KEYWORDS: list[str] = [
    "repay", "repayment", "repaid",
    "refinanc",
    "pay down", "paydown",
    "redeem", "redemption",
    "outstanding indebtedness",
    "credit facility", "credit agreement",
    "term loan", "revolving credit",
    "senior note", "senior secured",
    "subordinated",
    "discharge", "extinguish",
    "outstanding debt",
    "loan repayment",
    "retire our debt",
    "repurchase",
]

GROWTH_KEYWORDS: list[str] = [
    "capital expenditure", "capex",
    "research and development", "r&d",
    "acquisitions", "acquire",
    "business combinations",
    "expand", "expansion",
    "new products", "product development",
    "infrastructure",
    "technology investment",
    "hire", "headcount", "personnel",
    "sales and marketing",
    "geographic expansion", "international expansion",
    "new markets", "new geographies",
    "strategic investment",
    "invest in growth",
    "increase our capacity",
    "fund our growth",
]

GENERAL_KEYWORDS: list[str] = [
    "general corporate purposes",
    "working capital",
    "general and administrative",
    "operating expenses",
    "day-to-day operations",
    "overhead",
    "general business",
]

SECONDARY_KEYWORDS: list[str] = [
    "selling stockholder",
    "selling shareholder",
    "secondary offering",
    "existing stockholder",
    "existing shareholders",
    "proceeds to the selling",
    "no proceeds to the company",
    "we will not receive",
    "will not receive any proceeds",
    "proceeds will not be received",
    "registered for resale",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _keyword_score(text: str, keywords: list[str]) -> float:
    """Count how many distinct keyword phrases appear in text (case-insensitive)."""
    text_lower = text.lower()
    return float(sum(1 for kw in keywords if kw in text_lower))


def _locate_proceeds_text(sections: dict) -> tuple[str, int]:
    """
    Return (proceeds_text, section_found_flag).
    Tries dedicated section first, then searches summary for proceeds language.
    """
    proceeds_text = sections.get("use_of_proceeds", "").strip()
    if proceeds_text:
        return proceeds_text, 1

    # Fallback: locate "use of proceeds" heading inside summary
    summary = sections.get("summary", "")
    pattern = re.search(r"use\s+of\s+proceeds", summary, re.IGNORECASE)
    if pattern:
        start = pattern.start()
        return summary[start: start + 3000].strip(), 0

    return "", 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_proceeds_features(ticker: str, sections: dict) -> tuple[dict, str]:
    """
    Extract proceeds features for a single IPO.
    Returns (feature_dict, raw_text_snippet_for_debug).
    """
    proceeds_text, section_found = _locate_proceeds_text(sections)

    if not proceeds_text:
        logger.debug("No use-of-proceeds text for %s", ticker)
        return (
            {
                "ticker": ticker,
                "proceeds_section_found": 0,
                "proceeds_text_length": 0,
                "proceeds_debt_score": 0.0,
                "proceeds_growth_score": 0.0,
                "proceeds_general_score": 0.0,
                "proceeds_secondary_score": 0.0,
                "proceeds_debt_pct": None,
                "proceeds_growth_pct": None,
                "proceeds_general_pct": None,
                "proceeds_secondary_pct": None,
                "has_debt_repayment_flag": 0,
                "has_growth_flag": 0,
            },
            "",
        )

    debt_s    = _keyword_score(proceeds_text, DEBT_KEYWORDS)
    growth_s  = _keyword_score(proceeds_text, GROWTH_KEYWORDS)
    general_s = _keyword_score(proceeds_text, GENERAL_KEYWORDS)
    second_s  = _keyword_score(proceeds_text, SECONDARY_KEYWORDS)
    total     = debt_s + growth_s + general_s + second_s

    if total > 0:
        debt_pct    = debt_s    / total
        growth_pct  = growth_s  / total
        general_pct = general_s / total
        second_pct  = second_s  / total
    else:
        debt_pct = growth_pct = general_pct = second_pct = None

    features = {
        "ticker": ticker,
        "proceeds_section_found": section_found,
        "proceeds_text_length": len(proceeds_text),
        "proceeds_debt_score": debt_s,
        "proceeds_growth_score": growth_s,
        "proceeds_general_score": general_s,
        "proceeds_secondary_score": second_s,
        "proceeds_debt_pct": debt_pct,
        "proceeds_growth_pct": growth_pct,
        "proceeds_general_pct": general_pct,
        "proceeds_secondary_pct": second_pct,
        "has_debt_repayment_flag": int(debt_s > 0),
        "has_growth_flag": int(growth_s > 0),
    }
    return features, proceeds_text[:500]


def build_proceeds_features(
    sections_dir: Path = SECTIONS_DIR,
    output_path: Path = OUTPUT_PATH,
    debug_path: Path = DEBUG_PATH,
) -> pd.DataFrame:
    """Process all section JSONs and save proceeds features CSV."""
    section_files = sorted(sections_dir.glob("*.json"))
    if not section_files:
        raise FileNotFoundError(f"No section JSON files in {sections_dir}. Run section_extractor.py first.")

    print(f"Extracting proceeds features for {len(section_files)} IPOs...")
    records, debug_rows = [], []

    for path in section_files:
        try:
            sections = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path.name, exc)
            continue

        ticker = sections.get("ticker", path.stem)
        feat, snippet = extract_proceeds_features(ticker, sections)
        records.append(feat)
        debug_rows.append({"ticker": ticker, "proceeds_snippet": snippet})

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    pd.DataFrame(debug_rows).to_csv(debug_path, index=False)

    section_pct = df["proceeds_section_found"].mean()
    has_hits = (df[["proceeds_debt_score", "proceeds_growth_score",
                     "proceeds_general_score", "proceeds_secondary_score"]]
                .sum(axis=1) > 0).mean()

    print(f"\nSaved: {output_path}  ({len(df)} rows)")
    print(f"Debug snippets: {debug_path}")
    print(f"  Proceeds section found:  {section_pct:.0%}")
    print(f"  Any keyword hits:        {has_hits:.0%}")
    print(f"  has_debt_repayment_flag: {df['has_debt_repayment_flag'].mean():.1%}")
    print(f"  has_growth_flag:         {df['has_growth_flag'].mean():.1%}")
    return df


if __name__ == "__main__":
    build_proceeds_features()
