"""Shared Streamlit UI helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

SAMPLE_CSV = "qid,question\nQ1,What is the difference between INNER JOIN and LEFT JOIN?\nQ2,Write a query to find the second highest salary.\n"

STATUS_LABELS = {
    "stored": "Stored",
    "duplicate": "Duplicate",
    "new": "Unique",
    "already_stored": "Already stored",
    "already_exists": "Already in pool",
    "qid_conflict": "ID conflict",
    "borderline_stored": "Stored (borderline)",
}


def render_file_format_help() -> None:
    with st.expander("File format help", expanded=False):
        st.markdown(
            """
Upload a **CSV** or **XLSX** file with these columns:

| Column | Accepted names |
|--------|----------------|
| Question ID | `qid`, `question id`, `id` |
| Question text | `question`, `raw_question`, `text` |

Each row needs both a question ID and question text.
            """
        )
        st.download_button(
            label="Download sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_questions.csv",
            mime="text/csv",
        )


def render_upload_controls() -> tuple[Any | None, str | None]:
    """Render uploader and return (file object, filename)."""
    uploaded_file = st.file_uploader(
        "Upload a CSV or XLSX file",
        type=["csv", "xlsx"],
        help="Max 500 rows and 5 MB per upload.",
    )
    if uploaded_file is not None:
        st.session_state["uploaded_file_bytes"] = uploaded_file.getbuffer().tobytes()
        st.session_state["uploaded_file_name"] = uploaded_file.name

    file_name = st.session_state.get("uploaded_file_name")
    if file_name:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"Selected: **{file_name}**")
        with col2:
            if st.button("Clear", key=f"clear_{file_name}"):
                st.session_state.pop("uploaded_file_bytes", None)
                st.session_state.pop("uploaded_file_name", None)
                st.rerun()

    if uploaded_file is None and st.session_state.get("uploaded_file_bytes"):
        from io import BytesIO

        buffer = BytesIO(st.session_state["uploaded_file_bytes"])
        buffer.name = st.session_state["uploaded_file_name"]
        return buffer, st.session_state["uploaded_file_name"]

    return uploaded_file, file_name


def summarize_results(results: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        status = item.status if hasattr(item, "status") else item.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def render_result_summary(results: list[Any], *, action: str = "processed") -> None:
    if not results:
        return

    counts = summarize_results(results)
    labels = []
    for status, count in sorted(counts.items()):
        label = STATUS_LABELS.get(status, status.replace("_", " ").title())
        labels.append(f"**{label}:** {count}")
    st.info(f"{len(results)} question(s) {action} · " + " · ".join(labels))


def apply_status_colors(df: pd.DataFrame) -> Any:
    friendly_to_raw = {label.lower(): key for key, label in STATUS_LABELS.items()}

    def _row_style(row: pd.Series) -> list[str]:
        display_status = str(row.get("Status", "")).lower()
        status = friendly_to_raw.get(display_status, display_status)
        if status in {"duplicate", "qid_conflict"}:
            color = "background-color: #fde2e2"
        elif status in {"stored", "new", "borderline_stored"}:
            color = "background-color: #dcfce7"
        elif status in {"already_stored", "already_exists"}:
            color = "background-color: #e5e7eb"
        else:
            color = ""
        return [color] * len(row)

    status_col = "Status" if "Status" in df.columns else None
    if status_col is None:
        return df
    return df.style.apply(_row_style, axis=1)


def make_progress_callback(total: int, label: str) -> Callable[[int, int], None]:
    progress = st.progress(0.0, text=f"{label} 0/{total}")
    status = st.empty()

    def _update(done: int, _total: int) -> None:
        if total <= 0:
            return
        ratio = min(done / total, 1.0)
        progress.progress(ratio, text=f"{label} {done}/{total}")
        if done >= total:
            status.caption("Finishing up...")

    return _update
