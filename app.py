from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from common.auth import require_authentication
from common.config import get_settings
from common.logging_config import setup_logging
from common.ui import STATUS_LABELS, apply_status_colors

DEFAULT_SUBJECTS = ["sql", "python", "javascript", "react", "dsa"]

if TYPE_CHECKING:
    from agent2.storage import ChromaRAGStore, StoreResult

try:
    from agent1.agent import enrich_question
    from agent2.storage import ChromaRAGStore, StoreResult

    agent_import_error = None
except ImportError as exc:
    enrich_question = None
    ChromaRAGStore = None
    StoreResult = None
    agent_import_error = exc


def configure_page() -> None:
    st.set_page_config(
        page_title="Question Duplicate Checker",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_session_state() -> None:
    defaults = {
        "uploaded_file_bytes": None,
        "uploaded_file_name": None,
        "pending_duplicate": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_file_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def get_uploaded_file_source(uploaded_file):
    if uploaded_file is not None:
        file_bytes = uploaded_file.getbuffer().tobytes()
        st.session_state["uploaded_file_bytes"] = file_bytes
        st.session_state["uploaded_file_name"] = uploaded_file.name

    file_bytes = st.session_state.get("uploaded_file_bytes")
    file_name = st.session_state.get("uploaded_file_name")
    if file_bytes is None or file_name is None:
        return None, None

    buffer = BytesIO(file_bytes)
    buffer.name = file_name
    return buffer, file_name


def get_available_subjects() -> list[str]:
    subjects = set(DEFAULT_SUBJECTS)
    if agent_import_error is not None:
        return sorted(subjects)

    try:
        probe = ChromaRAGStore.for_subject(DEFAULT_SUBJECTS[0])
        subjects.update(probe.list_subjects())
    except Exception:
        pass

    return sorted(subjects)


def get_rag_store(subject: str, threshold: float) -> Any:
    if agent_import_error is not None:
        st.error(
            f"Backend imports failed: {agent_import_error}. "
            "Install dependencies and run again."
        )
        return None

    if "rag_stores" not in st.session_state:
        st.session_state["rag_stores"] = {}

    subject_key = subject.strip().lower()
    if subject_key not in st.session_state["rag_stores"]:
        st.session_state["rag_stores"][subject_key] = ChromaRAGStore.for_subject(subject_key)

    store = st.session_state["rag_stores"][subject_key]
    store.duplicate_distance_threshold = threshold
    return store


def get_pool_count(subject: str) -> int | None:
    try:
        store = get_rag_store(subject, get_settings().duplicate_distance_threshold)
        if store is None:
            return None
        return store.count_questions()
    except Exception:
        return None


def format_error(exc: Exception) -> str:
    lower_message = str(exc).lower()
    if "api_key" in lower_message or "credential" in lower_message:
        return "Missing or invalid API credentials. Check your API key and environment settings."
    if isinstance(exc, ValueError):
        return str(exc)
    network_indicators = ["network", "connection", "timeout", "unreachable"]
    if any(token in lower_message for token in network_indicators):
        return "Network error occurred. Check your internet connection and retry."
    return "An unexpected error occurred. Check logs for details."


def build_results_dataframe(rows: list[Any]) -> pd.DataFrame:
    results = []
    for item in rows:
        if isinstance(item, dict):
            similarity = item.get("similarity")
            qid = item.get("qid")
            status = item.get("status")
            original_qid = item.get("original_qid")
            message = item.get("message")
        else:
            similarity = getattr(item, "similarity", None)
            qid = item.qid
            status = item.status
            original_qid = item.original_qid
            message = item.message

        similarity_value = f"{similarity:.1f}%" if isinstance(similarity, (float, int)) else "—"
        results.append(
            {
                "Question ID": qid,
                "Status": STATUS_LABELS.get(status, status.replace("_", " ").title()),
                "Original QID": original_qid or "—",
                "Similarity": similarity_value,
                "Message": message or "",
            }
        )
    return pd.DataFrame(results)


def render_subject_selector() -> tuple[str, float]:
    st.header("Settings")
    available_subjects = get_available_subjects()
    selected_subject = st.selectbox(
        "Subject",
        options=available_subjects,
        index=0,
        help="Questions are stored in separate pools per subject.",
    )
    threshold = st.slider(
        "Duplicate sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("threshold", get_settings().duplicate_distance_threshold)),
        step=0.01,
        help="Lower = stricter matching. Try 0.20–0.30 for most subjects.",
    )

    pool_count = get_pool_count(selected_subject)
    if pool_count is not None:
        st.metric("Questions in pool", pool_count)

    return selected_subject, threshold


def process_manual_entry(
    question_id: str,
    question_text: str,
    subject: str,
    threshold: float,
    allow_duplicate: bool = False,
):
    if not question_id.strip():
        st.warning("Please enter a Question ID.")
        return None
    if not question_text.strip():
        st.warning("Please enter the question text.")
        return None

    rag_store = get_rag_store(subject, threshold)
    if rag_store is None:
        return None

    try:
        with st.spinner("Enriching and checking question..."):
            enriched = enrich_question(question_id.strip(), question_text.strip(), subject)
            store_result = rag_store.add_or_flag_duplicate(
                question_id.strip(),
                question_text.strip(),
                enriched.enriched_text,
                allow_duplicate=allow_duplicate,
            )
        return store_result
    except Exception as exc:
        st.error(f"Failed to process question: {format_error(exc)}")
        return None


def render_manual_entry_section(subject: str, threshold: float) -> None:
    question_id = st.text_input("Question ID", placeholder="e.g. Q101")
    question_text = st.text_area("Question", placeholder="Paste the full question text here...", height=150)

    col1, _col2 = st.columns([1, 3])
    with col1:
        submit = st.button("Store question", type="primary", width="stretch")

    pending = st.session_state.get("pending_duplicate")
    if pending is not None and pending.get("qid") != question_id.strip():
        st.session_state["pending_duplicate"] = None
        pending = None

    if submit:
        result = process_manual_entry(
            question_id, question_text, subject, threshold, allow_duplicate=False
        )
        if result is not None:
            if result.status == "duplicate":
                similarity_text = (
                    f" ({result.similarity:.1f}% similar)"
                    if isinstance(result.similarity, (float, int))
                    else ""
                )
                st.session_state["pending_duplicate"] = {
                    "qid": question_id.strip(),
                    "question_text": question_text.strip(),
                    "similarity_text": similarity_text,
                }
            elif result.status == "stored":
                st.session_state["pending_duplicate"] = None
                st.success("Question stored successfully.")
            elif result.status == "already_stored":
                st.session_state["pending_duplicate"] = None
                st.info("This question is already in the pool.")
            elif result.status == "qid_conflict":
                st.session_state["pending_duplicate"] = None
                st.error("This Question ID is already used for a different question.")
            else:
                st.session_state["pending_duplicate"] = None
                st.info(result.message or "Question processed.")

            if result.status != "duplicate":
                st.dataframe(apply_status_colors(build_results_dataframe([result])), width="stretch")

    pending = st.session_state.get("pending_duplicate")
    if pending is None:
        return

    similarity_text = pending.get("similarity_text", "")
    st.warning(f"Similar question already in the pool{similarity_text}.")
    col1, col2 = st.columns(2)
    with col1:
        store_anyway = st.button("Store anyway", type="primary")
    with col2:
        cancel = st.button("Cancel")

    if store_anyway:
        saved = process_manual_entry(
            pending["qid"],
            pending["question_text"],
            subject,
            threshold,
            allow_duplicate=True,
        )
        st.session_state["pending_duplicate"] = None
        if saved is not None:
            st.success("Duplicate stored.")
            st.dataframe(apply_status_colors(build_results_dataframe([saved])), width="stretch")
    elif cancel:
        st.session_state["pending_duplicate"] = None
        st.rerun()


def main() -> None:
    setup_logging()
    configure_page()
    init_session_state()

    if not require_authentication():
        return

    with st.sidebar:
        st.title("Question Dedup")
        st.caption("Enrich · Check · Store")
        subject, threshold = render_subject_selector()
        if get_settings().auth_enabled and st.button("Sign out", width="stretch"):
            st.session_state.pop("authenticated", None)
            st.rerun()

    st.session_state["subject"] = subject
    st.session_state["threshold"] = threshold

    pages = {
        "Workflow": [
            st.Page("pages/store_pool.py", title="Store Pool", icon="🗄️", default=True),
            st.Page("pages/check_pool.py", title="Check Pool", icon="🔍"),
        ]
    }
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
