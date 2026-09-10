"""Optional password gate for the Streamlit UI."""

from __future__ import annotations

import streamlit as st

from common.config import get_settings


def require_authentication() -> bool:
    """Return True when the user is authenticated or auth is disabled."""
    settings = get_settings()
    if not settings.auth_enabled:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("Sign in")
    st.caption("Authentication is required to access this application.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if username == settings.app_username and password == settings.app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Invalid username or password.")

    return False
