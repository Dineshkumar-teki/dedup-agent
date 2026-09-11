import pandas as pd
import streamlit as st

from agent1.agent import enrich_question
from app import (
    build_results_dataframe,
    format_error,
    get_rag_store,
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


def show() -> None:
    if "subject" not in st.session_state:
        st.session_state["subject"] = "sql"
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = float(get_settings().duplicate_distance_threshold)

    subject = st.session_state["subject"]
    threshold = st.session_state["threshold"]

    st.header("Check Pool")
    st.markdown(
        "Check whether questions already exist in the pool **without storing them**. "
        "Use this before adding new content to your question bank."
    )
    render_file_format_help()

    uploaded_file, file_name = render_upload_controls()

    tab_upload, tab_manual = st.tabs(["Batch check", "Single question"])
    with tab_upload:
        if st.button("Check for duplicates", type="primary", disabled=uploaded_file is None):
            if uploaded_file is None:
                st.warning("Please upload a file first.")
                st.stop()

            rag_store = get_rag_store(subject, threshold)
            if rag_store is None:
                st.stop()

            temp_path = None
            try:
                temp_path = save_uploaded_file_to_temp(uploaded_file)
                file_size = uploaded_file.getbuffer().nbytes
                rows = load_questions_from_file(str(temp_path), file_size=file_size)
                if not rows:
                    st.info("No valid rows found. Check your file format.")
                    st.stop()

                progress = make_progress_callback(len(rows), "Checking questions")
                enriched_rows, error_rows = enrich_rows_parallel(
                    rows,
                    subject,
                    on_progress=progress,
                )

                if error_rows:
                    st.warning(f"{len(error_rows)} question(s) could not be enriched.")
                    st.dataframe(
                        pd.DataFrame(error_rows)[["qid", "violations"]],
                        width="stretch",
                        hide_index=True,
                    )

                with st.spinner("Comparing against pool..."):
                    results = rag_store.check_duplicates_batch(enriched_rows)

                if results:
                    render_result_summary(results, action="checked")
                    st.dataframe(
                        apply_status_colors(build_results_dataframe(results)),
                        width="stretch",
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(f"Failed to process file: {format_error(exc)}")
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        elif file_name:
            st.caption("Click **Check for duplicates** when ready.")

    with tab_manual:
        question_id = st.text_input("Question ID", placeholder="Optional", key="check_qid")
        question_text = st.text_area(
            "Question",
            placeholder="Paste the question text to check...",
            height=150,
            key="check_question",
        )

        if st.button("Check question", type="primary", width="stretch"):
            if not question_text.strip():
                st.warning("Please enter the question text.")
                st.stop()

            rag_store = get_rag_store(subject, threshold)
            if rag_store is None:
                st.stop()

            qid_value = question_id.strip()
            display_qid = qid_value or "manual_check"

            try:
                with st.spinner("Checking question..."):
                    enriched = enrich_question(display_qid, question_text.strip(), subject)
                    result = rag_store.check_duplicate(qid_value or "", enriched.enriched_text)

                if result.status == "new":
                    st.success("Unique — this question is not in the pool.")
                elif result.status == "duplicate":
                    msg = "Duplicate found in the pool."
                    if isinstance(result.similarity, (float, int)):
                        msg += f" Similarity: {result.similarity:.1f}%."
                    if result.original_qid:
                        msg += f" Matches: {result.original_qid}."
                    st.warning(msg)
                elif result.status == "already_exists":
                    st.info("This exact question is already in the pool.")
                elif result.status == "qid_conflict":
                    st.error("Question ID conflict — this ID belongs to a different question.")
                else:
                    st.info(result.message or "Check complete.")

                st.dataframe(
                    apply_status_colors(build_results_dataframe([result])),
                    width="stretch",
                    hide_index=True,
                )
            except Exception as exc:
                st.error(f"Failed to check question: {format_error(exc)}")


show()
