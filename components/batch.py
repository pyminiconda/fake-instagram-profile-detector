"""
batch.py — Batch Analysis page.

Upload a CSV of usernames and analyze them all in sequence.
"""

import io
import csv
from datetime import datetime

import streamlit as st
import pandas as pd

from core.feature_extractor import FeatureExtractor
from core.prediction_engine import PredictionEngine
from core.instaloader_fetch import ProfileFetcher
from core.history_manager import HistoryManager
from core.report_generator import ReportGenerator


def show_batch_page(db, prediction_engine: PredictionEngine,
                    profile_fetcher: ProfileFetcher,
                    history_manager: HistoryManager):
    """Render the batch analysis page."""
    st.markdown("## 📦 Batch Profile Analysis")
    st.markdown("Upload a CSV file with Instagram usernames to analyze multiple profiles at once.")

    if not prediction_engine.is_ready():
        st.warning("⚠️ Model not loaded. Train the model first via Admin → Model Training.")
        st.stop()

    # ── File uploader ──
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain a column named 'username' with Instagram usernames.",
        key="batch_csv_upload",
    )

    if uploaded_file is None:
        st.info("📎 Upload a CSV file to get started. The file should have a column named **username**.")

        # Show sample format
        with st.expander("📋 Sample CSV format"):
            st.code("username\njohn_doe\njane_smith\ntest_account_123", language="csv")
        return

    # ── Parse CSV ──
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    # Find username column
    username_col = None
    for col in df.columns:
        if col.strip().lower() in ("username", "usernames", "user", "ig_username", "instagram"):
            username_col = col
            break

    if username_col is None:
        st.error(
            "❌ Could not find a 'username' column in the CSV. "
            "Please make sure your CSV has a column named 'username'."
        )
        return

    usernames = df[username_col].dropna().astype(str).str.strip().tolist()
    usernames = [u.lstrip("@") for u in usernames if u]

    if not usernames:
        st.error("No valid usernames found in the CSV.")
        return

    st.success(f"✅ Found **{len(usernames)}** usernames to analyze.")

    # ── Run analysis ──
    if st.button("🚀 Start Batch Analysis", type="primary", key="start_batch_btn"):
        extractor = FeatureExtractor()
        results = []
        errors = []

        progress_bar = st.progress(0, text="Starting batch analysis...")

        for i, username in enumerate(usernames):
            progress = (i + 1) / len(usernames)
            progress_bar.progress(progress, text=f"Analyzing @{username} ({i+1}/{len(usernames)})...")

            try:
                # Try to fetch profile
                if not profile_fetcher.is_demo_mode():
                    profile = profile_fetcher.fetch_profile(username)
                    if profile:
                        features = extractor.extract_from_profile(profile)
                    else:
                        # Demo mode fallback — skip this user
                        errors.append({"username": username, "error": "Could not fetch (demo mode)"})
                        continue
                else:
                    # In demo mode, we can't fetch — use placeholder features
                    # This is a limitation; batch in demo mode makes little sense
                    errors.append({"username": username, "error": "Demo mode — no live data"})
                    continue

                prediction = prediction_engine.predict(features)

                result = {
                    "username": username,
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "risk_level": "High" if prediction["label"] == "fake" else "Low",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                results.append(result)

                # Save to history
                history_manager.save_result(
                    user_id=st.session_state["user_id"],
                    username=username,
                    label=prediction["label"],
                    confidence=prediction["confidence"],
                    shap_values=prediction["shap_values"],
                )

            except Exception as e:
                errors.append({"username": username, "error": str(e)})

        progress_bar.progress(1.0, text="✅ Batch analysis complete!")

        # ── Display results ──
        if results:
            st.markdown("### 📊 Results")
            results_df = pd.DataFrame(results)
            results_df.columns = ["Username", "Label", "Confidence", "Risk Level", "Timestamp"]
            results_df["Confidence"] = results_df["Confidence"].apply(lambda x: f"{x:.1%}")

            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Summary stats
            fake_count = sum(1 for r in results if r["label"] == "fake")
            genuine_count = len(results) - fake_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Analyzed", len(results))
            col2.metric("Fake", fake_count)
            col3.metric("Genuine", genuine_count)

            # Download buttons
            st.markdown("### 📥 Download Results")
            col_a, col_b = st.columns(2)

            with col_a:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    data=csv_data,
                    file_name="batch_results.csv",
                    mime="text/csv",
                    key="batch_csv_download",
                )

            with col_b:
                gen = ReportGenerator()
                pdf_bytes = gen.generate_batch_report(
                    results=results,
                    generated_by=st.session_state.get("username", "System"),
                )
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name="batch_report.pdf",
                    mime="application/pdf",
                    key="batch_pdf_download",
                )

        # Show errors if any
        if errors:
            with st.expander(f"⚠️ {len(errors)} errors encountered"):
                for err in errors:
                    st.write(f"- **@{err['username']}**: {err['error']}")
