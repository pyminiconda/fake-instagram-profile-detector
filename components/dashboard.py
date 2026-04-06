"""
dashboard.py — Single Profile Analysis page.

Handles both live Instaloader mode and demo mode (manual feature input).
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.feature_extractor import FeatureExtractor, FEATURE_NAMES
from core.prediction_engine import PredictionEngine
from core.instaloader_fetch import (
    ProfileFetcher, ProfileNotFoundError, RateLimitError, PrivateProfileError,
)
from core.history_manager import HistoryManager
from core.report_generator import ReportGenerator


def show_dashboard(db, prediction_engine: PredictionEngine,
                   profile_fetcher: ProfileFetcher,
                   history_manager: HistoryManager):
    """Render the single profile analysis dashboard."""

    st.markdown("## 🔍 Profile Analysis Dashboard")
    st.markdown("Analyze an Instagram profile for authenticity.")

    if not prediction_engine.is_ready():
        st.warning(
            "⚠️ **Model not loaded.** Please ask an admin to train the model first "
            "(Admin → Model Training)."
        )
        st.stop()

    extractor = FeatureExtractor()

    # ── Always show both options as tabs ──
    tab_live, tab_manual = st.tabs(["🌐 Live Fetch (Username)", "📝 Manual Entry"])

    with tab_live:
        if profile_fetcher.is_demo_mode():
            st.info(
                "🌐 **Anonymous Mode** — No Instagram credentials found, but you can still "
                "fetch **public profiles** by entering a username below. "
                "For private profiles or higher rate limits, add `INSTA_USERNAME` and "
                "`INSTA_PASSWORD` to your `.env` file."
            )
        _show_live_input(extractor, prediction_engine, profile_fetcher,
                         history_manager, db)

    with tab_manual:
        st.info(
            "📝 **Manual Mode** — Enter profile statistics yourself for analysis."
        )
        _show_manual_input(extractor, prediction_engine, history_manager, db)


def _show_live_input(extractor, prediction_engine, profile_fetcher,
                     history_manager, db):
    """Live mode — fetch from Instagram."""
    col1, col2 = st.columns([3, 1])
    with col1:
        username = st.text_input(
            "Instagram Username",
            placeholder="Enter a public Instagram username",
            key="analysis_username",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔎 Analyze", type="primary",
                                use_container_width=True, key="analyze_btn")

    if analyze_btn and username:
        username = username.strip().lstrip("@")
        if not username:
            st.error("Please enter a valid username.")
            return

        with st.spinner(f"Fetching profile data for @{username}..."):
            try:
                profile = profile_fetcher.fetch_profile(username)
            except ProfileNotFoundError as e:
                st.error(f"❌ {e}")
                return
            except RateLimitError as e:
                st.warning(f"⏳ {e}")
                return
            except PrivateProfileError as e:
                st.warning(f"🔒 {e}")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        if profile is None:
            st.error(
                "❌ **Could not fetch profile.** Possible reasons:\n"
                "- Instagram may be blocking automated requests from this IP\n"
                "- The username may be incorrect\n"
                "- Try using the **Manual Entry** tab instead"
            )
            return

        # Extract features & predict
        features = extractor.extract_from_profile(profile)
        prediction = prediction_engine.predict(features)

        # Display results
        _display_profile_card(profile)
        _display_prediction(prediction)
        _display_risk_profile(prediction)
        _display_shap_chart(prediction)
        _display_download_button(profile, prediction, db)

        # Auto-save to history
        history_manager.save_result(
            user_id=st.session_state["user_id"],
            username=username,
            label=prediction["label"],
            confidence=prediction["confidence"],
            shap_values=prediction["shap_values"],
        )


def _show_manual_input(extractor, prediction_engine, history_manager, db):
    """Demo mode — manual feature input form."""
    with st.container(border=True):
        st.markdown("### 📝 Enter Profile Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            username = st.text_input("Username", value="test_user", key="manual_username")
            followers = st.number_input("Followers", min_value=0, value=100, key="manual_followers")
            has_pic = st.checkbox("Has Profile Picture", value=True, key="manual_pic")
        with col2:
            following = st.number_input("Following", min_value=0, value=200, key="manual_following")
            posts = st.number_input("Posts", min_value=0, value=10, key="manual_posts")
            has_url = st.checkbox("Has External URL", value=False, key="manual_url")
        with col3:
            bio_length = st.number_input("Bio Length (chars)", min_value=0, value=50, key="manual_bio")
            full_name = st.text_input("Full Name", value="", key="manual_name")

    analyze_btn = st.button("🔎 Analyze Profile", type="primary", key="manual_analyze_btn")

    if analyze_btn:
        features = extractor.extract_from_manual_input(
            followers=followers, following=following, posts=posts,
            has_pic=has_pic, bio_length=bio_length, username=username,
            has_url=has_url, full_name=full_name,
        )
        prediction = prediction_engine.predict(features)

        # Build a profile-like dict for display
        profile = {
            "username": username,
            "followersCount": followers,
            "followingCount": following,
            "postsCount": posts,
            "hasProfilePicture": has_pic,
            "isPrivate": False,
            "isVerified": False,
            "biography": "x" * bio_length,
            "externalUrl": "https://example.com" if has_url else "",
            "fullName": full_name,
        }

        _display_profile_card(profile)
        _display_prediction(prediction)
        _display_risk_profile(prediction)
        _display_shap_chart(prediction)
        _display_download_button(profile, prediction, db)

        # Auto-save
        history_manager.save_result(
            user_id=st.session_state["user_id"],
            username=username,
            label=prediction["label"],
            confidence=prediction["confidence"],
            shap_values=prediction["shap_values"],
        )


# ==================================================================
# Display helpers
# ==================================================================
def _display_profile_card(profile: dict):
    """Show the profile snapshot card."""
    st.markdown("---")
    st.markdown("### 👤 Profile Snapshot")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Followers", f"{profile.get('followersCount', 0):,}")
    col2.metric("Following", f"{profile.get('followingCount', 0):,}")
    col3.metric("Posts", f"{profile.get('postsCount', 0):,}")
    col4.metric("Profile Pic", "✅ Yes" if profile.get("hasProfilePicture") else "❌ No")

    with st.expander("More details"):
        st.write(f"**Username:** @{profile.get('username', 'N/A')}")
        st.write(f"**Full Name:** {profile.get('fullName', 'N/A')}")
        st.write(f"**Private:** {'Yes' if profile.get('isPrivate') else 'No'}")
        st.write(f"**Verified:** {'Yes' if profile.get('isVerified') else 'No'}")
        bio = profile.get("biography", "")
        st.write(f"**Bio:** {bio if bio else 'No bio'}")
        url = profile.get("externalUrl", "")
        st.write(f"**External URL:** {url if url else 'None'}")


def _display_prediction(prediction: dict):
    """Show the prediction result with label badge and confidence gauge."""
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    label = prediction["label"]
    confidence = prediction["confidence"]

    col1, col2 = st.columns(2)

    with col1:
        if label == "fake":
            st.markdown(
                '<div style="background: linear-gradient(135deg, #ff4444, #cc0000); '
                'color: white; padding: 1.5rem; border-radius: 12px; text-align: center; '
                'font-size: 1.8rem; font-weight: bold;">🚫 FAKE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #00C851, #007E33); '
                'color: white; padding: 1.5rem; border-radius: 12px; text-align: center; '
                'font-size: 1.8rem; font-weight: bold;">✅ GENUINE</div>',
                unsafe_allow_html=True,
            )

    with col2:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": "Confidence", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1a73e8"},
                "steps": [
                    {"range": [0, 50], "color": "#ffcccc"},
                    {"range": [50, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    if prediction["low_confidence"]:
        st.warning(
            "⚠️ **Low Confidence Warning** — The model's confidence is below 70%. "
            "This prediction may not be reliable. Consider verifying manually."
        )


def _display_risk_profile(prediction: dict):
    """Show the risk profile card with per-feature flags."""
    st.markdown("---")
    st.markdown("### 🛡️ Risk Profile")

    risk_flags = prediction.get("risk_flags", {})
    cols = st.columns(2)

    for i, (feature, info) in enumerate(risk_flags.items()):
        with cols[i % 2]:
            flag = info["flag"]
            note = info["note"]
            bg = "#fff3cd" if flag == "⚠️" else "#d4edda"
            border = "#ffc107" if flag == "⚠️" else "#28a745"
            st.markdown(
                f'<div style="background: {bg}; border-left: 4px solid {border}; '
                f'padding: 0.6rem 1rem; margin: 0.3rem 0; border-radius: 6px;">'
                f'<b>{flag} {feature}</b><br><small>{note}</small></div>',
                unsafe_allow_html=True,
            )


def _display_shap_chart(prediction: dict):
    """Show SHAP feature importance as a horizontal bar chart."""
    st.markdown("---")
    st.markdown("### 📊 Feature Importance (SHAP)")

    shap_values = prediction.get("shap_values", {})
    if not shap_values or all(v == 0 for v in shap_values.values()):
        st.info("SHAP values not available for this prediction.")
        return

    # Sort by absolute value
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]))
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#dc3545" if v > 0 else "#28a745" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title="SHAP Values — Impact on Prediction",
        xaxis_title="SHAP Value (→ Fake | ← Genuine)",
        height=350,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("What do SHAP values mean?"):
        st.markdown("""
        - **Red bars (positive)** push the prediction toward **FAKE**
        - **Green bars (negative)** push the prediction toward **GENUINE**
        - Longer bars = stronger influence on the prediction
        """)


def _display_download_button(profile: dict, prediction: dict, db):
    """Show PDF download button."""
    st.markdown("---")
    gen = ReportGenerator()
    pdf_bytes = gen.generate_single_report(
        profile=profile,
        prediction=prediction,
        generated_by=st.session_state.get("username", "System"),
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"analysis_{profile.get('username', 'profile')}.pdf",
        mime="application/pdf",
        key="download_pdf_btn",
    )
