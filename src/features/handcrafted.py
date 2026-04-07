"""
handcrafted.py
--------------
Compute handcrafted text features from IPO filing sections.

Features:
- Sentiment score (positive / negative tone via VADER)
- Uncertainty language density
- Risk section length (absolute + relative)
- Profitability vs loss framing ratio
- Growth keyword frequency
- Insider selling signal (keyword-based)
- Readability score (Flesch-Kincaid approximation)
- Forward-looking statement density
"""

import re
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import (
    PROCESSED_DIR, UNCERTAINTY_KEYWORDS, GROWTH_KEYWORDS,
    PROFIT_KEYWORDS, LOSS_KEYWORDS
)

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("NLTK VADER not available — sentiment will be keyword-based only")

SECTIONS_DIR = PROCESSED_DIR / "sections"
OUTPUT_PATH = PROCESSED_DIR / "handcrafted_features.csv"

INSIDER_KEYWORDS = [
    "selling stockholder", "insider", "secondary offering",
    "existing stockholder", "selling shareholders",
]

FORWARD_LOOKING_KEYWORDS = [
    "we believe", "we expect", "we anticipate", "we intend", "we plan",
    "we estimate", "we project", "going forward", "in the future",
]


def keyword_density(text: str, keywords: list[str]) -> float:
    """Fraction of words that match keywords."""
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(kw in w for kw in keywords))
    return hits / len(words)


def phrase_density(text: str, phrases: list[str]) -> float:
    """Occurrences of multi-word phrases per 1000 words."""
    if not text:
        return 0.0
    text_lower = text.lower()
    word_count = len(re.findall(r"\b\w+\b", text_lower))
    if word_count == 0:
        return 0.0
    hits = sum(text_lower.count(phrase) for phrase in phrases)
    return (hits / word_count) * 1000


def sentiment_score(text: str) -> dict[str, float]:
    """Return VADER compound score + positive/negative breakdown."""
    if not text:
        return {"sentiment_compound": 0.0, "sentiment_pos": 0.0, "sentiment_neg": 0.0}

    if VADER_AVAILABLE:
        sia = SentimentIntensityAnalyzer()
        # Score first 5000 chars (VADER is slow on long text)
        scores = sia.polarity_scores(text[:5000])
        return {
            "sentiment_compound": scores["compound"],
            "sentiment_pos": scores["pos"],
            "sentiment_neg": scores["neg"],
        }
    else:
        # Fallback: simple positive/negative word count ratio
        words = re.findall(r"\b\w+\b", text.lower())
        pos = sum(1 for w in words if w in PROFIT_KEYWORDS + GROWTH_KEYWORDS)
        neg = sum(1 for w in words if w in LOSS_KEYWORDS + UNCERTAINTY_KEYWORDS)
        total = max(len(words), 1)
        return {
            "sentiment_compound": (pos - neg) / total,
            "sentiment_pos": pos / total,
            "sentiment_neg": neg / total,
        }


def flesch_reading_ease(text: str) -> float:
    """Approximate Flesch reading ease score."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text)
    syllables = sum(count_syllables(w) for w in words)

    if not sentences or not words:
        return 50.0  # neutral default

    asl = len(words) / len(sentences)  # avg sentence length
    asw = syllables / len(words)       # avg syllables per word

    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return round(max(0, min(100, score)), 2)


def count_syllables(word: str) -> int:
    """Rough syllable count."""
    word = word.lower()
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)


def extract_features_from_sections(sections: dict) -> dict:
    """
    Compute all handcrafted features from a filing's extracted sections.
    """
    # Combine all text and individual sections
    section_texts = {k: v for k, v in sections.items()
                     if isinstance(v, str) and k not in ["ticker", "filing_type", "filing_date", "error"]}
    full_text = " ".join(section_texts.values())

    risk_text = sections.get("risk_factors", "")
    summary_text = sections.get("summary", "")
    business_text = sections.get("business", "")
    proceeds_text = sections.get("use_of_proceeds", "")

    features = {}

    # --- Document structure ---
    features["total_text_length"] = len(full_text)
    features["risk_section_length"] = len(risk_text)
    features["risk_to_total_ratio"] = len(risk_text) / max(len(full_text), 1)
    features["n_sections_found"] = len(section_texts)

    # --- Sentiment (on summary + business) ---
    analysis_text = summary_text + " " + business_text
    sent = sentiment_score(analysis_text)
    features.update(sent)

    # --- Uncertainty / hedging language ---
    features["uncertainty_density"] = keyword_density(full_text, UNCERTAINTY_KEYWORDS)
    features["uncertainty_in_risk"] = keyword_density(risk_text, UNCERTAINTY_KEYWORDS)

    # --- Growth vs profitability framing ---
    features["growth_keyword_density"] = keyword_density(full_text, GROWTH_KEYWORDS)
    features["profit_keyword_density"] = keyword_density(full_text, PROFIT_KEYWORDS)
    features["loss_keyword_density"] = keyword_density(full_text, LOSS_KEYWORDS)
    features["profit_loss_ratio"] = (
        features["profit_keyword_density"] /
        max(features["loss_keyword_density"], 0.0001)
    )

    # --- Forward-looking statements ---
    features["forward_looking_density"] = phrase_density(full_text, FORWARD_LOOKING_KEYWORDS)

    # --- Insider selling signal ---
    features["insider_selling_signal"] = int(
        any(kw in full_text.lower() for kw in INSIDER_KEYWORDS)
    )

    # --- Readability ---
    features["readability_score"] = flesch_reading_ease(summary_text or full_text[:3000])

    # --- Risk factor count (rough) ---
    risk_bullets = len(re.findall(r"\n[•\-\*]|\n\d+\.", risk_text))
    features["risk_factor_count_approx"] = risk_bullets

    return features


def build_feature_matrix(sections_dir: Path = SECTIONS_DIR) -> pd.DataFrame:
    """Process all section JSONs and build feature matrix."""
    records = []
    json_files = list(sections_dir.glob("*.json"))
    print(f"Processing {len(json_files)} section files...")

    for fpath in json_files:
        with open(fpath) as f:
            sections = json.load(f)

        ticker = sections.get("ticker", fpath.stem)
        features = extract_features_from_sections(sections)
        features["ticker"] = ticker
        features["filing_type"] = sections.get("filing_type", "unknown")
        records.append(features)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved handcrafted features: {OUTPUT_PATH} ({len(df)} rows, {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    df = build_feature_matrix()
    if df.empty:
        print("No section files found — run section_extractor.py first")
    else:
        print(df.describe().T.to_string())
