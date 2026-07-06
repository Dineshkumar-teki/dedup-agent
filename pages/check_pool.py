import pandas as pd
import streamlit as st

from app import (
    get_rag_store,
    render_manual_entry_section,
)
from agent1.agent import enrich_question
from common.io import load_questions_from_file


def highlight_check_status(value: object) -> str:
    status = str(value).lower()
    if status in {"duplicate", "already_exists", "qid_conflict"}:
        return "color: #ff4444"
    if status == "unique":
        return "color: #00cc44"
    return ""


def render_file_upload_section(subject: str, threshold: float) -> None:
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.session_state["uploaded_file_bytes"] = uploaded_file.getbuffer().tobytes()
        st.session_state["uploaded_file_name"] = uploaded_file.name

    if st.session_state.get("uploaded_file_name"):
        st.write(f"Uploaded file: {st.session_state['uploaded_file_name']}")

    status_message = st.empty()
    result_area = st.empty()

    if st.button("Process File"):
        if uploaded_file is None:
            st.warning("Please upload a file before processing.")
            st.stop()

        rag_store = get_rag_store(subject, threshold)
        if rag_store is None:
            return

        status_message.info("Checking questions...")
        try:
            temp_path = None
            try:
                from app import save_uploaded_file_to_temp

                temp_path = save_uploaded_file_to_temp(uploaded_file)
                rows = load_questions_from_file(str(temp_path))
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

            results = []
            for qid, raw_question in rows:
                enriched = enrich_question(str(qid), str(raw_question), subject)
                result = rag_store.check_duplicate(str(qid), enriched.enriched_text)
                results.append(result)

            if results:
                df_results = pd.DataFrame(
                    [
                        {
                            "Question ID": item.qid,
                            "Status": item.status,
                            "Original QID": item.original_qid,
                            "Similarity": f"{item.similarity:.1f}%" if isinstance(item.similarity, (float, int)) else None,
                            "Message": item.message,
                        }
                        for item in results
                    ]
                )
                status_message.success("Check completed.")
                result_area.dataframe(df_results.style.map(highlight_check_status, subset=["Status"]))
            else:
                status_message.info("No valid rows found in file.")
        except Exception as exc:
            status_message.error(f"Failed to process the file: {exc}")

    if st.session_state.get("last_results"):
        if st.session_state.get("last_status") == "success":
            status_message.success(
                "Showing last successful result from {}.".format(st.session_state.get("last_results_source"))
            )
            result_area.dataframe(pd.DataFrame(st.session_state["last_results"]).style.map(highlight_check_status, subset=["Status"]))
        elif st.session_state.get("last_status") == "error":
            status_message.error("Last operation failed: {}".format(st.session_state.get("last_error")))


def render_manual_entry_section(subject: str, threshold: float) -> None:
    st.header("Manual Question Entry")
    question_id = st.text_input("Question ID")
    question_text = st.text_area("Question")

    status_message = st.empty()
    result_area = st.empty()

    if st.button("Check Question"):
        if not question_id.strip():
            st.warning("Question ID is empty")
            return
        if not question_text.strip():
            st.warning("Question text is empty")
            return

        rag_store = get_rag_store(subject, threshold)
        if rag_store is None:
            return

        status_message.info("Checking question...")
        try:
            enriched = enrich_question(question_id.strip(), question_text.strip(), subject)
            result = rag_store.check_duplicate(question_id.strip(), enriched.enriched_text)

            if result.status == "unique":
                st.success("Question is unique. Not in pool.")
            elif result.status == "duplicate":
                st.warning("Duplicate found.")
                if result.original_qid:
                    st.write(f"Duplicate of: {result.original_qid}")
                if isinstance(result.similarity, (float, int)):
                    st.write(f"Similarity: {result.similarity:.1f}%")
            elif result.status == "already_exists":
                st.info("Question already exists in pool.")
            elif result.status == "qid_conflict":
                st.error("QID conflict — same ID exists with different content.")
            else:
                st.info(result.message or "No duplicate found.")

            result_area.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Question ID": result.qid,
                            "Status": result.status,
                            "Original QID": result.original_qid,
                            "Similarity": f"{result.similarity:.1f}%" if isinstance(result.similarity, (float, int)) else None,
                            "Message": result.message,
                        }
                    ]
                )
            )
        except Exception as exc:
            status_message.error(f"Failed to check question: {exc}")


def show() -> None:
    subject = st.session_state["subject"]
    threshold = st.session_state["threshold"]

    tab_upload, tab_manual = st.tabs(["File Upload", "Manual Entry"])
    with tab_upload:
        render_file_upload_section(subject, threshold)
    with tab_manual:
        render_manual_entry_section(subject, threshold)


show()
