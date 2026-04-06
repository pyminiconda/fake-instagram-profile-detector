"""
auth.py — Login and Signup pages for the Streamlit app.

Uses bcrypt for password hashing and st.session_state for session management.
"""

import re
import streamlit as st
from core.database import DatabaseManager


def show_login_page(db: DatabaseManager):
    """Display the login form."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1a73e8; font-size: 2.5rem;">🔍 Fake Profile Detector</h1>
        <p style="color: #666; font-size: 1.1rem;">Instagram Profile Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("### 🔐 Login")

            email = st.text_input("Email", placeholder="your@email.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")

            col_a, col_b = st.columns(2)
            with col_a:
                login_btn = st.button("Login", type="primary", use_container_width=True, key="login_btn")
            with col_b:
                signup_btn = st.button("Create Account", use_container_width=True, key="goto_signup_btn")

            if login_btn:
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = db.authenticate_user(email, password)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user_id"] = user["userId"]
                        st.session_state["username"] = user["username"]
                        st.session_state["email"] = user["email"]
                        st.session_state["is_admin"] = bool(user.get("is_admin", 0))
                        st.session_state["page"] = "dashboard"
                        st.success(f"Welcome back, {user['username']}! 🎉")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

            if signup_btn:
                st.session_state["page"] = "signup"
                st.rerun()


def show_signup_page(db: DatabaseManager):
    """Display the signup form."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1a73e8; font-size: 2.5rem;">🔍 Fake Profile Detector</h1>
        <p style="color: #666; font-size: 1.1rem;">Create your account</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("### ✏️ Sign Up")

            username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
            email = st.text_input("Email", placeholder="your@email.com", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="signup_confirm")

            col_a, col_b = st.columns(2)
            with col_a:
                signup_btn = st.button("Sign Up", type="primary", use_container_width=True, key="signup_btn")
            with col_b:
                back_btn = st.button("Back to Login", use_container_width=True, key="goto_login_btn")

            if signup_btn:
                # Validations
                errors = []
                if not username or not email or not password:
                    errors.append("All fields are required.")
                if username and len(username) < 3:
                    errors.append("Username must be at least 3 characters.")
                if username and len(username) > 30:
                    errors.append("Username must be 30 characters or fewer.")
                if email and not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
                    errors.append("Invalid email format.")
                if password and len(password) < 6:
                    errors.append("Password must be at least 6 characters.")
                if password != confirm_password:
                    errors.append("Passwords do not match.")

                # Check duplicates
                if not errors:
                    dupes = db.check_duplicate(username=username, email=email)
                    if dupes["username_exists"]:
                        errors.append("Username already taken.")
                    if dupes["email_exists"]:
                        errors.append("Email already registered.")

                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    try:
                        db.create_user(username, email, password)
                        st.success("Account created successfully! 🎉 Please login.")
                        st.session_state["page"] = "login"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")

            if back_btn:
                st.session_state["page"] = "login"
                st.rerun()


def logout():
    """Clear session and redirect to login."""
    for key in ["authenticated", "user_id", "username", "email", "is_admin"]:
        st.session_state.pop(key, None)
    st.session_state["page"] = "login"
    st.rerun()
