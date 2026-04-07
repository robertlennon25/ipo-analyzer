"""
streamlit_app.py
----------------
MVP UI for the IPO Language & Aftermarket Performance Analyzer.

Pages:
1. IPO Explorer — browse companies, filing text, return metrics
2. Model Results — compare M1/M2/M3 performance
3. Feature Analysis — top predictive features, SHAP summary
"""

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, MODELS_DIR if Path("config/settings.py").exists() else Path("data/processed")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO Signal Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_returns():
    path = PROCESSED_DIR / "returns.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_handcrafted():
    path = PROCESSED_DIR / "handcrafted_features.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_model_results():
    path = PROCESSED_DIR / "model_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_sections(ticker: str):
    import glob
    files = list((PROCESSED_DIR / "sections").glob(f"{ticker}_*.json"))
    if files:
        with open(files[0]) as f:
            return json.load(f)
    return {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 IPO Signal Analyzer")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 IPO Explorer", "🤖 Model Results", "📊 Feature Analysis"],
)

returns_df = load_returns()
features_df = load_handcrafted()
model_results = load_model_results()

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("IPO Language & Aftermarket Performance")
    st.markdown("""
    **Core Hypothesis**: Language in IPO filings (S-1 / 424B4) contains measurable signals
    that correlate with aftermarket stock performance.
    
    This system ingests SEC filings, extracts text features and financial multiples,
    and trains models to predict post-IPO returns at 1D, 1W, 1M, 6M, and 1Y horizons.
    """)

    if returns_df.empty:
        st.warning("No data loaded yet. Run the ingestion pipeline first.")
        st.code("python src/ingestion/ipo_list.py\npython src/ingestion/edgar_fetcher.py\npython src/ingestion/price_fetcher.py")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("IPOs in Universe", len(returns_df))
        with col2:
            if "ret_1m" in returns_df.columns:
                pos_rate = (returns_df["ret_1m"] > 0).mean()
                st.metric("1M Positive Rate", f"{pos_rate:.0%}")
        with col3:
            if "ret_1m" in returns_df.columns:
                med = returns_df["ret_1m"].median()
                st.metric("Median 1M Return", f"{med:.1%}")
        with col4:
            n_sectors = returns_df["sector"].nunique() if "sector" in returns_df.columns else 0
            st.metric("Sectors", n_sectors)

        # Return distribution
        if "ret_1m" in returns_df.columns:
            st.subheader("1-Month Return Distribution")
            fig = px.histogram(
                returns_df.dropna(subset=["ret_1m"]),
                x="ret_1m",
                nbins=40,
                title="Distribution of 1-Month Post-IPO Returns",
                labels={"ret_1m": "1-Month Return"},
                color_discrete_sequence=["#2563EB"],
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# ── Page: IPO Explorer ────────────────────────────────────────────────────────
elif page == "🔍 IPO Explorer":
    st.title("IPO Explorer")

    if returns_df.empty:
        st.warning("No return data available. Run the pipeline first.")
    else:
        # Filters
        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("Search by ticker or company", "")
        with col2:
            if "sector" in returns_df.columns:
                sectors = ["All"] + sorted(returns_df["sector"].dropna().unique().tolist())
                sector_filter = st.selectbox("Sector", sectors)

        filtered = returns_df.copy()
        if search:
            mask = (
                filtered["ticker"].str.contains(search.upper(), na=False) |
                filtered.get("company", pd.Series()).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if "sector" in returns_df.columns and sector_filter != "All":
            filtered = filtered[filtered["sector"] == sector_filter]

        # Table
        display_cols = ["ticker", "company", "ipo_date", "offer_price", "sector",
                        "ret_1d", "ret_1w", "ret_1m", "ret_6m", "ret_1y"]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].style.format({
                c: "{:.1%}" for c in display_cols if c.startswith("ret_")
            }),
            use_container_width=True,
            height=400,
        )

        # Detail view
        st.subheader("Filing Text Explorer")
        if not filtered.empty:
            selected = st.selectbox("Select IPO", filtered["ticker"].tolist())
            sections = load_sections(selected)

            if sections:
                for section_name in ["summary", "risk_factors", "business", "use_of_proceeds"]:
                    text = sections.get(section_name, "")
                    if text and len(text) > 50:
                        with st.expander(f"📄 {section_name.replace('_', ' ').title()} ({len(text):,} chars)"):
                            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))
            else:
                st.info(f"No parsed sections found for {selected}. Run the parsing pipeline.")

# ── Page: Model Results ───────────────────────────────────────────────────────
elif page == "🤖 Model Results":
    st.title("Model Comparison")
    st.markdown("""
    | Variant | Features | Hypothesis |
    |---------|----------|------------|
    | **M1** | Text (NLP + embeddings) | Pure language signal |
    | **M2** | Multiples + deal structure | Pure fundamentals signal |
    | **M3** | M1 + M2 combined | Does text add alpha over numbers? |
    """)

    if model_results is None:
        st.warning("No model results found. Run `python src/modeling/train.py` first.")
    else:
        target = model_results.get("target", "unknown")
        st.info(f"Target variable: **{target}**")

        # Build comparison table
        rows = []
        for variant, models in model_results.get("variants", {}).items():
            for model_type, metrics in models.items():
                rows.append({
                    "Variant": variant,
                    "Model": model_type,
                    "ROC-AUC": metrics["roc_auc_mean"],
                    "AUC Std": metrics["roc_auc_std"],
                    "Accuracy": metrics["accuracy_mean"],
                    "N Samples": metrics["n_samples"],
                })

        comparison_df = pd.DataFrame(rows)

        # Bar chart
        fig = px.bar(
            comparison_df,
            x="Variant",
            y="ROC-AUC",
            color="Model",
            barmode="group",
            error_y="AUC Std",
            title="ROC-AUC by Model Variant",
            color_discrete_map={"logistic_regression": "#2563EB", "xgboost": "#16A34A"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Random baseline")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comparison_df.round(3), use_container_width=True)

# ── Page: Feature Analysis ────────────────────────────────────────────────────
elif page == "📊 Feature Analysis":
    st.title("Feature Analysis")

    if model_results is None:
        st.warning("Run the modeling pipeline first.")
    else:
        # Feature importance for each variant
        for variant, models in model_results.get("variants", {}).items():
            st.subheader(f"{variant} — Top Features")
            xgb_results = models.get("xgboost", {})
            top_features = xgb_results.get("top_features", [])

            if top_features:
                feat_df = pd.DataFrame(top_features[:15])
                fig = px.bar(
                    feat_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"{variant} (XGBoost) — Feature Importance",
                    color="importance",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature importance data available for this variant.")

    # Raw feature distributions
    if not features_df.empty:
        st.subheader("Handcrafted Feature Distributions")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "ticker"]

        selected_feature = st.selectbox("Feature", numeric_cols)

        if not returns_df.empty and "label_1m" in returns_df.columns:
            merged = features_df.merge(returns_df[["ticker", "label_1m"]], on="ticker", how="inner")
            merged["performance"] = merged["label_1m"].map({1: "Positive 1M", 0: "Negative 1M"})
            fig = px.histogram(
                merged.dropna(subset=[selected_feature]),
                x=selected_feature,
                color="performance",
                barmode="overlay",
                opacity=0.7,
                title=f"{selected_feature} by 1-Month Performance",
                color_discrete_map={"Positive 1M": "#16A34A", "Negative 1M": "#DC2626"},
            )
            st.plotly_chart(fig, use_container_width=True)
