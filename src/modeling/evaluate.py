"""
evaluate.py
-----------
Standalone evaluation module for trained IPO analyzer models.

1. Loads trained models from data/processed/models/
2. Loads model_results.json
3. Generates SHAP summary plots (saved as PNGs to data/processed/plots/)
4. Produces a Markdown evaluation report (data/processed/evaluation_report.md)

Report includes:
- Model comparison table (ROC-AUC, accuracy, n_samples)
- Top 10 features per variant
- Interpretation: does text add signal over fundamentals?
- Notable finding callouts
"""

import sys
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, RANDOM_STATE, CV_FOLDS  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODELS_DIR = PROCESSED_DIR / "models"
PLOTS_DIR = PROCESSED_DIR / "plots"
RESULTS_PATH = PROCESSED_DIR / "model_results.json"
REPORT_PATH = PROCESSED_DIR / "evaluation_report.md"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"model_results.json not found at {RESULTS_PATH}. Run train.py first."
        )
    with open(RESULTS_PATH) as f:
        return json.load(f)


def load_feature_matrix(variant_name: str, target: str) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    """
    Reconstruct the feature matrix for a variant so SHAP can be computed.
    Returns (X, feature_names, y) or None if data is unavailable.
    """
    try:
        from src.features.embeddings import load_embeddings, embeddings_to_dataframe
    except ImportError:
        load_embeddings = None

    hc_path = PROCESSED_DIR / "handcrafted_features.csv"
    multiples_path = PROCESSED_DIR / "multiples_features.csv"
    returns_path = PROCESSED_DIR / "returns.csv"

    if not returns_path.exists():
        logger.warning("returns.csv not found — cannot reconstruct feature matrices")
        return None

    returns_df = pd.read_csv(returns_path)
    if target not in returns_df.columns:
        logger.warning("Target %s not in returns.csv", target)
        return None

    market_path = PROCESSED_DIR / "market_context_features.csv"
    parts: list[pd.DataFrame] = []

    if "M1" in variant_name or "M3" in variant_name:
        if hc_path.exists():
            hc = pd.read_csv(hc_path)
            drop = ["filing_type"] + [c for c in hc.columns if hc[c].dtype == object and c != "ticker"]
            parts.append(hc.drop(columns=[c for c in drop if c in hc.columns]))
        if load_embeddings is not None:
            try:
                embeddings, tickers = load_embeddings()
                parts.append(embeddings_to_dataframe(embeddings, tickers))
            except FileNotFoundError:
                pass

    if "M2" in variant_name or "M3" in variant_name:
        if multiples_path.exists():
            mult = pd.read_csv(multiples_path)
            drop = [c for c in mult.columns if mult[c].dtype == object and c != "ticker"]
            parts.append(mult.drop(columns=[c for c in drop if c in mult.columns]))
        if market_path.exists():
            mkt = pd.read_csv(market_path)
            drop = [c for c in mkt.columns if mkt[c].dtype == object and c != "ticker"]
            parts.append(mkt.drop(columns=[c for c in drop if c in mkt.columns]))

    if not parts:
        return None

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="ticker", how="inner")

    merged = merged.merge(returns_df[["ticker", target]], on="ticker", how="inner")
    merged = merged.dropna(subset=[target])

    if len(merged) < 10:
        logger.warning("Only %d samples for %s — skipping SHAP", len(merged), variant_name)
        return None

    y = merged[target].values.astype(int)
    feature_cols = [
        c for c in merged.columns
        if c not in ("ticker", target) and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    X = merged[feature_cols].values

    # Impute NaNs (same strategy as train.py)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    return X, feature_cols, y


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------

def generate_shap_plot(
    model_path: Path,
    X: np.ndarray,
    feature_names: list[str],
    variant_name: str,
    model_type: str,
) -> Path | None:
    """Generate and save a SHAP summary plot for one model. Returns plot path or None."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plots")
        return None

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.warning("Could not load model %s: %s", model_path, e)
        return None

    # Extract classifier and pre-transform X through all pipeline steps except the final clf
    if hasattr(model, "named_steps"):
        clf = model[-1]
        X_transformed = model[:-1].transform(X)
    else:
        clf = model
        X_transformed = X

    try:
        if hasattr(clf, "feature_importances_"):
            # Tree-based: use TreeExplainer (fast + exact)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
            # For binary XGB: shap_values is already 2D (n_samples × n_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Linear: use LinearExplainer or masker-based
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        logger.warning("SHAP explainer failed for %s/%s: %s", variant_name, model_type, e)
        return None

    # Plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"shap_{variant_name}_{model_type}.png"

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary — {variant_name} / {model_type}", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    logger.info("SHAP plot saved: %s", plot_path)
    return plot_path


# ---------------------------------------------------------------------------
# Report generation helpers
# ---------------------------------------------------------------------------

def _comparison_table(results: dict) -> str:
    """Build the model comparison Markdown table."""
    naive = None
    for models in results["variants"].values():
        for m in models.values():
            naive = m.get("naive_accuracy")
            break
        if naive:
            break

    rows = [
        f"Naive accuracy baseline: **{naive:.1%}**\n" if naive else "",
        "| Variant | Model | ROC-AUC | ± | Bal-Accuracy | ± | Pred +% | N |",
        "|---------|-------|---------|---|-------------|---|---------|---|",
    ]
    for variant, models in results["variants"].items():
        for model_name, m in models.items():
            rows.append(
                f"| {variant} | {model_name} "
                f"| {m['roc_auc_mean']:.3f} | {m['roc_auc_std']:.3f} "
                f"| {m.get('bal_accuracy_mean', m['accuracy_mean']):.3f} "
                f"| {m.get('bal_accuracy_std', m['accuracy_std']):.3f} "
                f"| {m.get('pred_positive_pct', '—')}% "
                f"| {m['n_samples']} |"
            )
    return "\n".join(rows)


def _top_features_section(results: dict) -> str:
    """Build top-10 features per variant section."""
    lines = []
    for variant, models in results["variants"].items():
        lines.append(f"\n### {variant}\n")
        for model_name, m in models.items():
            top = m.get("top_features", [])[:10]
            if not top:
                lines.append(f"**{model_name}**: no feature importance available\n")
                continue
            lines.append(f"**{model_name}** — top {len(top)} features:\n")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|------|---------|------------|")
            for i, feat in enumerate(top, 1):
                lines.append(f"| {i} | `{feat['feature']}` | {feat['importance']:.4f} |")
            lines.append("")
    return "\n".join(lines)


def _text_vs_fundamentals_interpretation(results: dict) -> str:
    """
    Compare M1 (text) vs M2 (fundamentals) vs M3 (combined) for the best model type.
    Returns an interpretation paragraph.
    """
    variants = results.get("variants", {})
    model_types = ["xgboost", "logistic_regression"]

    # Collect best AUC per variant
    aucs: dict[str, dict[str, float]] = {}
    for variant, models in variants.items():
        aucs[variant] = {
            mt: models[mt]["roc_auc_mean"]
            for mt in model_types
            if mt in models
        }

    lines = []

    for mt in model_types:
        m1 = aucs.get("M1_text", {}).get(mt)
        m2 = aucs.get("M2_multiples", {}).get(mt)
        m3 = aucs.get("M3_combined", {}).get(mt)

        if None in (m1, m2, m3):
            continue

        lines.append(f"\n**{mt}:**\n")
        lines.append(f"- M1 (text only): ROC-AUC = {m1:.3f}")
        lines.append(f"- M2 (fundamentals only): ROC-AUC = {m2:.3f}")
        lines.append(f"- M3 (combined): ROC-AUC = {m3:.3f}")

        text_delta = m3 - m2
        fund_delta = m3 - m1

        if text_delta > 0.01:
            lines.append(
                f"\n**Text adds signal:** M3 outperforms M2 by {text_delta:+.3f} AUC, "
                f"suggesting filing language contains predictive information beyond financials."
            )
        elif text_delta < -0.01:
            lines.append(
                f"\n**Text does not help:** M3 underperforms M2 by {abs(text_delta):.3f} AUC, "
                f"suggesting text may be adding noise or overfitting with this sample size."
            )
        else:
            lines.append(
                f"\n**Mixed result:** Text adds marginal lift ({text_delta:+.3f} AUC). "
                f"With a larger corpus the signal may become clearer."
            )

        best = max(m1, m2, m3)
        baseline = 0.5
        if best > 0.65:
            lines.append(
                f"Best variant achieves {best:.3f} AUC — meaningfully above the 0.5 random baseline."
            )
        elif best > 0.55:
            lines.append(
                f"Best variant achieves {best:.3f} AUC — modest lift above random. "
                f"Consider expanding the IPO universe for stronger signal."
            )
        else:
            lines.append(
                f"Best variant achieves {best:.3f} AUC — near random. "
                f"This is expected with a small sample; run with 200+ IPOs for meaningful results."
            )

    return "\n".join(lines) if lines else "_Not enough variants to compare._"


def _notable_findings(results: dict) -> str:
    """Surface the single most predictive feature across all variants."""
    findings = []
    seen_features: set[str] = set()

    for variant, models in results.get("variants", {}).items():
        for model_name, m in models.items():
            top = m.get("top_features", [])
            if not top:
                continue
            best_feat = top[0]
            fname = best_feat["feature"]
            if fname not in seen_features:
                seen_features.add(fname)
                findings.append(
                    f'- **`{fname}`** is the #1 feature in **{variant}/{model_name}** '
                    f'(importance: {best_feat["importance"]:.4f})'
                )

    # Check for specific interesting patterns
    all_top_features = []
    for variant, models in results.get("variants", {}).items():
        for model_name, m in models.items():
            all_top_features.extend(f["feature"] for f in m.get("top_features", [])[:5])

    uncertainty_count = sum(1 for f in all_top_features if "uncertainty" in f.lower())
    if uncertainty_count >= 2:
        findings.append(
            f"- Uncertainty/hedging language features appear in the top-5 of "
            f"{uncertainty_count} model variants — consistent with prior academic findings."
        )

    sentiment_count = sum(1 for f in all_top_features if "sentiment" in f.lower())
    if sentiment_count >= 2:
        findings.append(
            f"- Sentiment features appear in the top-5 of {sentiment_count} variants."
        )

    return "\n".join(findings) if findings else "_Run with more data to surface notable patterns._"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_report(results: dict, shap_plot_paths: dict[str, Path]) -> str:
    """Build the full Markdown evaluation report."""
    target = results.get("target", "unknown")
    n_variants = len(results.get("variants", {}))

    shap_section = ""
    if shap_plot_paths:
        shap_section = "\n## SHAP Feature Importance Plots\n\n"
        for name, path in shap_plot_paths.items():
            rel = path.relative_to(PROCESSED_DIR)
            shap_section += f"### {name}\n\n![SHAP]({rel})\n\n"

    report = f"""# IPO Language & Aftermarket Performance — Evaluation Report

**Target variable:** `{target}` (binary: did the stock outperform median 1-month return?)
**Variants evaluated:** {n_variants}
**Cross-validation:** {CV_FOLDS}-fold stratified

---

## Model Comparison

{_comparison_table(results)}

---

## Top Features by Variant

{_top_features_section(results)}

---

## Does Text Add Signal Over Fundamentals?

**Research question:** Does language in IPO filings (S-1 / 424B4) predict post-IPO returns
beyond what can be explained by structured financials alone?

{_text_vs_fundamentals_interpretation(results)}

---

## Notable Findings

{_notable_findings(results)}

---
{shap_section}
## Methodology Notes

- **M1:** Text-only features — VADER sentiment, uncertainty keyword density, readability,
  forward-looking statement density, and sentence-transformer embeddings (all-MiniLM-L6-v2).
- **M2:** Structured financial features extracted from filing HTML — revenue growth, gross margin,
  profitability, cash burn, total proceeds, sector.
- **M3:** M1 + M2 combined — tests whether text and fundamentals are complementary.
- All variants use 5-fold stratified cross-validation. Means and standard deviations are reported
  to guard against overfitting on the small IPO sample.
- **Sample size caveat:** With <100 IPOs, results are noisy. Run the full pipeline with 200–500
  IPOs (via `data/raw/ipo_list_override.csv`) for reliable conclusions.
"""
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """Full evaluation pipeline: SHAP plots + Markdown report for all targets."""
    all_results = load_results()

    # Support both old single-target format and new multi-target format
    if "variants" in all_results:
        all_results = {all_results["target"]: all_results}

    print(f"Loaded results for {len(all_results)} target(s): {list(all_results.keys())}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    report_sections: list[str] = [
        "# IPO Language & Aftermarket Performance — Evaluation Report\n",
        f"**Cross-validation:** {CV_FOLDS}-fold stratified  \n",
    ]

    # Stdout summary header
    print(f"\n{'='*80}")
    print(f"{'Variant':<20} {'Model':<22} {'Target':<12} {'ROC-AUC':<16} {'Bal-Acc':<16} {'Naive':<8} Pred +%")
    print(f"{'-'*80}")

    for target, results in all_results.items():
        shap_plot_paths: dict[str, Path] = {}

        for variant_name, models in results.get("variants", {}).items():
            data = load_feature_matrix(variant_name, target)

            for model_type in models:
                model_path = MODELS_DIR / f"{target}_{variant_name}_{model_type}.pkl"
                if not model_path.exists():
                    # Fall back to old naming convention
                    model_path = MODELS_DIR / f"{variant_name}_{model_type}.pkl"
                if not model_path.exists():
                    logger.warning("Model file not found: %s", model_path.name)
                    continue

                if data is not None:
                    X, feature_names, y = data
                    plot_path = generate_shap_plot(
                        model_path, X, feature_names,
                        f"{target}_{variant_name}", model_type
                    )
                    if plot_path:
                        shap_plot_paths[f"{variant_name}/{model_type}"] = plot_path

        # Per-target report section
        report_sections.append(f"\n---\n\n## Target: `{target}`\n")
        report_sections.append(_comparison_table(results))
        report_sections.append("\n### Top Features\n")
        report_sections.append(_top_features_section(results))
        report_sections.append("\n### Signal Interpretation\n")
        report_sections.append(_text_vs_fundamentals_interpretation(results))
        report_sections.append("\n### Notable Findings\n")
        report_sections.append(_notable_findings(results))

        if shap_plot_paths:
            report_sections.append("\n### SHAP Plots\n")
            for name, path in shap_plot_paths.items():
                rel = path.relative_to(PROCESSED_DIR)
                report_sections.append(f"**{name}**\n\n![SHAP]({rel})\n")

        # Stdout table rows
        for variant_name, models in results.get("variants", {}).items():
            for model_name, m in models.items():
                naive = m.get("naive_accuracy", 0)
                print(
                    f"{variant_name:<20} {model_name:<22} {target:<12} "
                    f"{m['roc_auc_mean']:.3f}±{m['roc_auc_std']:.3f}    "
                    f"{m.get('bal_accuracy_mean', m['accuracy_mean']):.3f}±"
                    f"{m.get('bal_accuracy_std', m['accuracy_std']):.3f}    "
                    f"{naive:.3f}   "
                    f"{m.get('pred_positive_pct', '—')}%"
                )

    report_md = "\n".join(report_sections)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")

    _append_to_results_tracker(all_results)


def _append_to_results_tracker(all_results: dict) -> None:
    """Append a run summary block to results_tracker.md."""
    from datetime import datetime
    tracker_path = Path(__file__).parent.parent.parent / "results_tracker.md"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"\n---\n\n## Auto-logged — {now}\n"]

    for target, results in all_results.items():
        naive = None
        for models in results.get("variants", {}).values():
            for m in models.values():
                naive = m.get("naive_accuracy")
                break
            if naive:
                break

        lines.append(f"\n### {target}  (naive acc: {naive:.1%})\n" if naive else f"\n### {target}\n")
        lines.append("| Variant | Model | ROC-AUC | Bal-Acc | Pred +% |")
        lines.append("|---------|-------|---------|---------|---------|")

        for variant, models in results.get("variants", {}).items():
            for model_name, m in models.items():
                lines.append(
                    f"| {variant} | {model_name} "
                    f"| {m['roc_auc_mean']:.3f} ± {m['roc_auc_std']:.3f} "
                    f"| {m.get('bal_accuracy_mean', m['accuracy_mean']):.3f} ± {m.get('bal_accuracy_std', m['accuracy_std']):.3f} "
                    f"| {m.get('pred_positive_pct', '—')}% |"
                )

    block = "\n".join(lines) + "\n"

    with open(tracker_path, "a") as f:
        f.write(block)

    print(f"Results appended to {tracker_path.name}")


if __name__ == "__main__":
    run_evaluation()
