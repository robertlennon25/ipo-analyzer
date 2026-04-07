"""
embeddings.py
-------------
Generate sentence-transformer embeddings for IPO filing sections.

Strategy:
- Embed each section independently (risk_factors, summary, business)
- Concatenate or average section embeddings per document
- Cache embeddings to avoid recomputation

Output: data/cache/embeddings.npz + data/processed/embedding_index.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CACHE_DIR, EMBEDDING_MODEL

SECTIONS_DIR = PROCESSED_DIR / "sections"
EMBED_CACHE = CACHE_DIR / "embeddings.npz"
INDEX_PATH = PROCESSED_DIR / "embedding_index.csv"

# Which sections to embed (in priority order)
SECTIONS_TO_EMBED = ["summary", "risk_factors", "business", "use_of_proceeds"]
MAX_EMBED_CHARS = 4000  # Truncate before embedding


def load_model():
    """Load sentence transformer model (cached after first load)."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_sections(sections: dict, model) -> np.ndarray:
    """
    Generate a single document embedding by:
    1. Embedding each available section
    2. Averaging across sections (weighted: summary > risk > business > proceeds)
    """
    weights = {
        "summary": 1.5,
        "risk_factors": 1.2,
        "business": 1.0,
        "use_of_proceeds": 0.8,
        "mda": 0.7,
    }

    section_embeddings = []
    section_weights = []

    for section_name in SECTIONS_TO_EMBED:
        text = sections.get(section_name, "")
        if not text or len(text) < 50:
            continue

        # Truncate
        text = text[:MAX_EMBED_CHARS]

        emb = model.encode(text, normalize_embeddings=True)
        section_embeddings.append(emb)
        section_weights.append(weights.get(section_name, 1.0))

    if not section_embeddings:
        # Fall back to full text snippet
        full = sections.get("full_text_fallback", "")
        if full:
            return model.encode(full[:MAX_EMBED_CHARS], normalize_embeddings=True)
        else:
            dim = model.get_sentence_embedding_dimension()
            return np.zeros(dim)

    # Weighted average
    weights_arr = np.array(section_weights)
    weights_arr = weights_arr / weights_arr.sum()
    stacked = np.stack(section_embeddings)
    return (stacked * weights_arr[:, None]).sum(axis=0)


def build_embeddings(sections_dir: Path = SECTIONS_DIR, force_recompute: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for all filings. Returns (embedding_matrix, ticker_list).
    """
    json_files = sorted(sections_dir.glob("*.json"))
    tickers = []

    # Load cached embeddings
    if EMBED_CACHE.exists() and not force_recompute:
        cached = np.load(EMBED_CACHE, allow_pickle=True)
        cached_tickers = list(cached["tickers"])
        cached_embeddings = cached["embeddings"]
        print(f"Loaded {len(cached_tickers)} cached embeddings")
    else:
        cached_tickers = []
        cached_embeddings = np.array([])

    # Find which tickers need new embeddings
    cached_set = set(cached_tickers)
    to_process = [f for f in json_files
                  if json.loads(f.read_text()).get("ticker", f.stem) not in cached_set]

    if not to_process:
        print("All embeddings up to date")
        return cached_embeddings, cached_tickers

    print(f"Computing embeddings for {len(to_process)} new filings...")
    model = load_model()

    new_tickers = []
    new_embeddings = []

    for fpath in to_process:
        with open(fpath) as f:
            sections = json.load(f)
        ticker = sections.get("ticker", fpath.stem)
        print(f"  Embedding: {ticker}")
        emb = embed_sections(sections, model)
        new_tickers.append(ticker)
        new_embeddings.append(emb)

    # Combine with cache
    if cached_embeddings.size > 0 and new_embeddings:
        all_embeddings = np.vstack([cached_embeddings, np.stack(new_embeddings)])
        all_tickers = cached_tickers + new_tickers
    elif new_embeddings:
        all_embeddings = np.stack(new_embeddings)
        all_tickers = new_tickers
    else:
        all_embeddings = cached_embeddings
        all_tickers = cached_tickers

    # Save updated cache
    np.savez(EMBED_CACHE, embeddings=all_embeddings, tickers=np.array(all_tickers))
    print(f"Embeddings saved: {all_embeddings.shape} ({EMBED_CACHE})")

    # Save index CSV for easy joining
    index_df = pd.DataFrame({
        "ticker": all_tickers,
        "embedding_idx": range(len(all_tickers)),
        "embedding_dim": all_embeddings.shape[1],
    })
    index_df.to_csv(INDEX_PATH, index=False)

    return all_embeddings, all_tickers


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load saved embeddings."""
    if not EMBED_CACHE.exists():
        raise FileNotFoundError(f"No embeddings cache found at {EMBED_CACHE}. Run build_embeddings() first.")
    cached = np.load(EMBED_CACHE, allow_pickle=True)
    return cached["embeddings"], list(cached["tickers"])


def embeddings_to_dataframe(embeddings: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    """Convert embedding matrix to DataFrame with ticker index."""
    dim = embeddings.shape[1]
    col_names = [f"emb_{i:03d}" for i in range(dim)]
    df = pd.DataFrame(embeddings, columns=col_names)
    df.insert(0, "ticker", tickers)
    return df


if __name__ == "__main__":
    embeddings, tickers = build_embeddings()
    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Sample tickers: {tickers[:5]}")
