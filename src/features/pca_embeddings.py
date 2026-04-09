"""
pca_embeddings.py
-----------------
Compress 384-dim sentence-transformer embeddings to N principal components.

PCA is fit on all available embeddings (unsupervised — no label information
leaks). The components are saved as a flat feature CSV that plugs directly
into the train_experiment.py feature pipeline.

Why this helps:
  - Random Forest splits on one feature per node. With 384 correlated dims,
    the first few dominate and the rest go unused. With 30 PCA components,
    variance is spread more evenly → RF explores more directions.
  - Reduces overfitting on small datasets (560 IPOs vs 384 feature dims).

Output:
    data/processed/pca_embeddings.csv        (ticker, pca_000 … pca_{N-1})
    data/processed/pca_embeddings_meta.json  (variance explained per component)

Usage:
    python src/features/pca_embeddings.py                   # 30 components (default)
    python src/features/pca_embeddings.py --n-components 50
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, CACHE_DIR  # noqa: E402

EMBED_CACHE = CACHE_DIR / "embeddings.npz"
DEFAULT_N_COMPONENTS = 30


def run(n_components: int = DEFAULT_N_COMPONENTS) -> None:
    if not EMBED_CACHE.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBED_CACHE}. Run embeddings.py first."
        )

    data = np.load(EMBED_CACHE, allow_pickle=True)
    embeddings: np.ndarray = data["embeddings"]   # (N, 384)
    tickers: list[str] = list(data["tickers"])

    print(f"Loaded embeddings: {embeddings.shape}  ({len(tickers)} IPOs)")

    # Standardize before PCA so no single dimension dominates due to scale.
    # Sentence embeddings are L2-normalised but dims vary in variance across
    # the corpus; StandardScaler makes PCA more evenly exploratory.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    n_components = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_per_comp = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_per_comp)

    print(f"PCA: {embeddings.shape[1]}d → {n_components}d")
    print(f"Cumulative variance explained: {cumvar[-1]:.1%}")
    print(f"\nVariance per component (first 15):")
    for i in range(min(15, n_components)):
        bar = "█" * int(var_per_comp[i] * 300)
        print(f"  PC{i+1:02d}: {var_per_comp[i]:.3%}  cumul={cumvar[i]:.3%}  {bar}")

    # Save feature CSV
    col_names = [f"pca_{i:03d}" for i in range(n_components)]
    df = pd.DataFrame(X_pca, columns=col_names)
    df.insert(0, "ticker", tickers)

    out_path = PROCESSED_DIR / "pca_embeddings.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({df.shape[0]} rows × {n_components} PCA features)")

    # Save metadata for reproducibility
    meta = {
        "n_components": n_components,
        "original_dim": int(embeddings.shape[1]),
        "n_samples": len(tickers),
        "variance_explained_total": float(cumvar[-1]),
        "variance_per_component": [float(v) for v in var_per_comp],
        "cumulative_variance": [float(v) for v in cumvar],
    }
    meta_path = PROCESSED_DIR / "pca_embeddings_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA-compress sentence-transformer embeddings")
    parser.add_argument(
        "--n-components", type=int, default=DEFAULT_N_COMPONENTS,
        help=f"Number of PCA components to retain (default: {DEFAULT_N_COMPONENTS}).",
    )
    args = parser.parse_args()
    run(args.n_components)
