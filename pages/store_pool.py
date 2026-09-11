import pandas as pd
import streamlit as st

from app import (
    build_results_dataframe,
    format_error,
    get_rag_store,
    render_manual_entry_section,
    save_uploaded_file_to_temp,
)
from common.config import get_settings
from common.enrichment import enrich_rows_parallel
from common.io import load_questions_from_file
from common.ui import (
    apply_status_colors,
    make_progress_callback,
    render_file_format_help,
    render_result_summary,
    render_upload_controls,
)
from common.validation import validate_enriched_text


def process_file(uploaded_file, subject: str, threshold: float):
    rag_store = get_rag_store(subject, threshold)
    if rag_store is None:
        return [], []

    temp_path = None
    try:
        temp_path = save_uploaded_file_to_temp(uploaded_file)
        file_size = uploaded_file.getbuffer().nbytes
        rows = load_questions_from_file(str(temp_path), file_size=file_size)
        if not rows:
            return [], []

        progress = make_progress_callback(len(rows), "Enriching questions")
        enriched_rows, exception_rows = enrich_rows_parallel(
            rows,
            subject,
            on_progress=progress,
        )

        clean_rows = []
        for qid, raw_question, enriched_text in enriched_rows:
            violations = validate_enriched_text(qid, enriched_text, subject)
            if violations:
                exception_rows.append(
                    {
                        "qid": qid,
                        "raw_question": raw_question,
                        "enriched_text": enriched_text,
                        "violations": ", ".join(violations),
                    }
                )
            else:
                clean_rows.append((qid, raw_question, enriched_text))

        if clean_rows:
            with st.spinner("Storing questions in pool..."):
                store_results = rag_store.batch_store(clean_rows)
        else:
            store_results = []

        return store_results, exception_rows
    except Exception as exc:
        st.error(f"Failed to process file: {format_error(exc)}")
        return [], []
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def show() -> None:
    if "subject" not in st.session_state:
        st.session_state["subject"] = "sql"
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = float(get_settings().duplicate_distance_threshold)

    subject = st.session_state["subject"]
    threshold = st.session_state["threshold"]

    st.header("Store Pool")
    st.markdown(
        "Upload questions to **enrich**, **validate**, and **store** them. "
        "Duplicates are flagged automatically — only clean questions are saved."
    )
    render_file_format_help()

    uploaded_file, file_name = render_upload_controls()

    tab_upload, tab_manual = st.tabs(["Batch upload", "Single question"])
    with tab_upload:
        if st.button("Process and store", type="primary", disabled=uploaded_file is None):
            if uploaded_file is None:
                st.warning("Please upload a file first.")
                st.stop()

            store_results, exception_rows = process_file(uploaded_file, subject, threshold)

            if not store_results and not exception_rows:
                st.info("No valid rows found. Check your file format.")
            if store_results:
                render_result_summary(store_results, action="stored")
                st.dataframe(
                    apply_status_colors(build_results_dataframe(store_results)),
                    width="stretch",
                    hide_index=True,
                )
            if exception_rows:
                st.warning(f"{len(exception_rows)} question(s) skipped due to errors.")
                st.dataframe(
                    pd.DataFrame(exception_rows)[["qid", "violations"]],
                    width="stretch",
                    hide_index=True,
                )
        elif file_name:
            st.caption("Click **Process and store** when ready.")

    with tab_manual:
        render_manual_entry_section(subject, threshold)


show()
