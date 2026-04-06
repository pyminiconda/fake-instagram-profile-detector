"""
Fake Instagram Profile Detection System — Main Streamlit Application.

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.database import DatabaseManager
from core.prediction_engine import PredictionEngine
from core.instaloader_fetch import ProfileFetcher
from core.history_manager import HistoryManager

from components.auth import show_login_page, show_signup_page, logout
from components.dashboard import show_dashboard
from components.batch import show_batch_page
from components.history import show_history_page
from components.dataset_insights import show_dataset_insights
from components.model_training import show_model_training


# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake Instagram Profile Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Container borders */
    [data-testid="stContainer"] {
        border-radius: 12px;
    }

    /* Hide default Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Sidebar nav items */
    .nav-item {
        padding: 0.6rem 1rem;
        margin: 0.2rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.2s;
    }
    .nav-item:hover {
        background: #dee2e6;
    }
    .nav-item.active {
        background: #1a73e8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Initialize services
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_db():
    return DatabaseManager()

@st.cache_resource
def init_prediction_engine():
    return PredictionEngine()

@st.cache_resource
def init_profile_fetcher(_db):
    return ProfileFetcher(_db)

def init_history_manager(db):
    return HistoryManager(db)


# ══════════════════════════════════════════════════════════════════════
# Session state defaults
# ══════════════════════════════════════════════════════════════════════
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"


# ══════════════════════════════════════════════════════════════════════
# Main app logic
# ══════════════════════════════════════════════════════════════════════
db = init_db()

# --- Not authenticated: show login/signup ---
if not st.session_state["authenticated"]:
    if st.session_state["page"] == "signup":
        show_signup_page(db)
    else:
        show_login_page(db)
    st.stop()

# --- Authenticated: show main app ---
prediction_engine = init_prediction_engine()
profile_fetcher = init_profile_fetcher(db)
history_manager = init_history_manager(db)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1a73e8; margin: 0;">🔍 FPD System</h2>
        <p style="color: #666; font-size: 0.85rem; margin: 0.3rem 0 0 0;">Fake Profile Detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # User info
    st.markdown(f"👤 **{st.session_state.get('username', 'User')}**")
    if st.session_state.get("is_admin"):
        st.markdown("🛡️ *Administrator*")

    st.markdown("---")

    # Navigation
    st.markdown("### Navigation")

    if st.button("🔍 Dashboard", use_container_width=True, key="nav_dashboard"):
        st.session_state["page"] = "dashboard"
        st.rerun()

    if st.button("📦 Batch Analysis", use_container_width=True, key="nav_batch"):
        st.session_state["page"] = "batch"
        st.rerun()

    if st.button("📜 Search History", use_container_width=True, key="nav_history"):
        st.session_state["page"] = "history"
        st.rerun()

    # Admin-only pages
    if st.session_state.get("is_admin"):
        st.markdown("---")
        st.markdown("### Admin")

        if st.button("📊 Dataset Insights", use_container_width=True, key="nav_insights"):
            st.session_state["page"] = "dataset_insights"
            st.rerun()

        if st.button("🧠 Model Training", use_container_width=True, key="nav_training"):
            st.session_state["page"] = "model_training"
            st.rerun()

    st.markdown("---")

    if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
        logout()


# ── Page routing ──
page = st.session_state.get("page", "dashboard")

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 1.8rem;">🔍 Fake Instagram Profile Detection System</h1>
    <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem;">
        AI-powered analysis to identify fake Instagram profiles
    </p>
</div>
""", unsafe_allow_html=True)

if page == "dashboard":
    show_dashboard(db, prediction_engine, profile_fetcher, history_manager)
elif page == "batch":
    show_batch_page(db, prediction_engine, profile_fetcher, history_manager)
elif page == "history":
    show_history_page(history_manager)
elif page == "dataset_insights":
    show_dataset_insights(db)
elif page == "model_training":
    show_model_training(db)
else:
    show_dashboard(db, prediction_engine, profile_fetcher, history_manager)
