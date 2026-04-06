"""
dataset_insights.py — Admin-only Dataset Insight Dashboard.

Visualizes dataset statistics, feature distributions, correlations,
model comparisons, confusion matrices, and ROC curves.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from ml.data_manager import DataManager
from ml.preprocessor import Preprocessor
from core.feature_extractor import FEATURE_NAMES


def show_dataset_insights(db):
    """Render the dataset insight dashboard (admin only)."""

    if not st.session_state.get("is_admin"):
        st.error("🔒 This page is restricted to administrators.")
        st.stop()

    st.markdown("## 📊 Dataset Insight Dashboard")
    st.markdown("Explore the InstaFake training dataset and model performance.")

    # ── Load dataset ──
    dm = DataManager()
    preprocessor = Preprocessor()

    try:
        with st.spinner("Loading dataset..."):
            df = dm.load_dataset()
            df_clean = dm.clean()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    info = dm.get_info()
    label_col = info["label_column"]

    # ── Section A: Dataset Overview ──
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{info['rows']:,}")
    col2.metric("Features", info['columns'] - 1)
    col3.metric("Label Column", label_col)

    # Class distribution
    st.markdown("#### Class Distribution")
    dist = info["label_distribution"]
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pie = px.pie(
            names=["Genuine (0)", "Fake (1)"],
            values=[dist.get(0, 0), dist.get(1, 0)],
            color_discrete_sequence=["#28a745", "#dc3545"],
            title="Label Distribution",
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(
            x=["Genuine (0)", "Fake (1)"],
            y=[dist.get(0, 0), dist.get(1, 0)],
            color=["Genuine", "Fake"],
            color_discrete_map={"Genuine": "#28a745", "Fake": "#dc3545"},
            title="Label Counts",
        )
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Check imbalance
    total = sum(dist.values())
    minority_pct = min(dist.values()) / total * 100
    if minority_pct < 40:
        st.warning(f"⚠️ Dataset is imbalanced — minority class is {minority_pct:.1f}% of total.")
    else:
        st.success(f"✅ Dataset is balanced — minority class is {minority_pct:.1f}% of total.")

    # ── Section B: Feature Distributions ──
    st.markdown("---")
    st.markdown("### 📈 Feature Distributions")

    # Engineer features for visualization
    try:
        X_features, y_labels = preprocessor.engineer_features(df_clean, label_col)
        viz_df = X_features.copy()
        viz_df["label"] = y_labels.map({0: "Genuine", 1: "Fake"})
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        return

    for feature in FEATURE_NAMES:
        with st.expander(f"📊 {feature}"):
            col1, col2 = st.columns(2)

            with col1:
                fig_hist = px.histogram(
                    viz_df, x=feature, color="label",
                    color_discrete_map={"Genuine": "#28a745", "Fake": "#dc3545"},
                    barmode="overlay", opacity=0.7,
                    title=f"{feature} — Histogram",
                    marginal="rug",
                )
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                fig_box = px.box(
                    viz_df, x="label", y=feature, color="label",
                    color_discrete_map={"Genuine": "#28a745", "Fake": "#dc3545"},
                    title=f"{feature} — Box Plot",
                )
                fig_box.update_layout(height=350)
                st.plotly_chart(fig_box, use_container_width=True)

            # Mean values
            means = viz_df.groupby("label")[feature].mean()
            st.write(f"**Mean values:** Genuine = {means.get('Genuine', 0):.4f}, "
                     f"Fake = {means.get('Fake', 0):.4f}")

    # ── Section C: Correlation Heatmap ──
    st.markdown("---")
    st.markdown("### 🔗 Correlation Heatmap")

    corr_df = X_features.copy()
    corr_df["is_fake"] = y_labels
    corr_matrix = corr_df.corr()

    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, linewidths=0.5,
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    st.pyplot(fig_corr)

    # Top correlated with label
    label_corr = corr_matrix["is_fake"].drop("is_fake").abs().sort_values(ascending=False)
    st.markdown("**Top 3 features most correlated with label:**")
    for i, (feat, val) in enumerate(label_corr.head(3).items()):
        st.write(f"  {i+1}. **{feat}** — |r| = {val:.4f}")

    # ── Section D: Most Predictive Features ──
    st.markdown("---")
    st.markdown("### 🎯 Most Predictive Features")
    st.info("Train the models using the Model Training page to see feature importances from Random Forest.")

    # ── Section E-G: Model Comparison (if models exist) ──
    all_models = db.get_all_models()
    if all_models:
        st.markdown("---")
        st.markdown("### 🏆 Model Comparison")

        model_df = pd.DataFrame(all_models)
        display_cols = ["algorithmType", "accuracy", "precision_score", "recall", "f1Score", "aucRoc", "isBestModel"]
        available_cols = [c for c in display_cols if c in model_df.columns]
        st.dataframe(model_df[available_cols], use_container_width=True, hide_index=True)
    else:
        st.markdown("---")
        st.info("💡 Train models via the **Model Training** page to see comparison metrics, confusion matrices, and ROC curves here.")
