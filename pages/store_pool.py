import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from app import render_manual_entry_section, get_rag_store, enrich_question, get_uploaded_file_source, save_uploaded_file_to_temp, build_results_dataframe, format_error
from common.io import load_questions_from_file


SQL_SYNTAX_TOKENS = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY"]

BATCH_SIZE = 10

CODE_SYNTAX_TOKENS = ["def ", "const ", "let ", "var ", "function ", "class ", "=>"]


def validate_enriched_text(qid: str, enriched_text: str, subject: str) -> list[str]:
    violations = []

    if "[LOW_CONFIDENCE]" in enriched_text:
        violations.append("LOW_CONFIDENCE flag present")

    if qid in enriched_text:
        violations.append("qid leaked into enriched_text")

    word_count = len(enriched_text.split())
    if word_count > 80:
        violations.append(f"enriched_text too long ({word_count} words)")

    if subject == "sql":
        tokens = SQL_SYNTAX_TOKENS
    else:
        tokens = CODE_SYNTAX_TOKENS

    if any(token in enriched_text for token in tokens):
        violations.append("code syntax detected in enriched_text")

    return violations


def enrich_single(index: int, qid: str, raw_question: str, subject: str) -> tuple:
    try:
        enriched = enrich_question(qid, raw_question, subject)
        return (index, qid, raw_question, enriched.enriched_text, None)
    except Exception as e:
        return (index, qid, raw_question, None, str(e))


def process_file(uploaded_file, subject: str, threshold: float):
    upload_source, file_name = get_uploaded_file_source(uploaded_file)
    if upload_source is None:
        return [], []

    rag_store = get_rag_store(subject, threshold)
    if rag_store is None:
        return [], []

    temp_path = None
    try:
        temp_path = save_uploaded_file_to_temp(upload_source)
        rows = load_questions_from_file(str(temp_path))

        enriched_rows = []
        exception_rows = []

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {
                executor.submit(enrich_single, idx, qid, raw_question, subject): idx
                for idx, (qid, raw_question) in enumerate(rows)
            }
            for future in as_completed(futures):
                index, qid, raw_question, enriched_text, error = future.result()
                if error is not None:
                    exception_rows.append({
                        "qid": qid,
                        "raw_question": raw_question,
                        "enriched_text": "",
                        "violations": f"Enrichment failed: {error}",
                    })
                else:
                    enriched_rows.append((index, qid, raw_question, enriched_text))

        enriched_rows.sort(key=lambda x: x[0])
        enriched_rows = [
            (qid, raw_question, enriched_text)
            for _, qid, raw_question, enriched_text in enriched_rows
        ]

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

        store_results = rag_store.batch_store(clean_rows)

        return store_results, exception_rows
    except Exception as exc:
        message = format_error(exc)
        st.error(f"Failed to process file: {message}")
        return [], []
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def render_file_upload_section(subject: str, threshold: float) -> None:
    st.header("File Upload")
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        st.session_state["uploaded_file_bytes"] = uploaded_file.getbuffer().tobytes()
        st.session_state["uploaded_file_name"] = uploaded_file.name

    if st.session_state.get("uploaded_file_name"):
        st.write(f"Uploaded file: {st.session_state['uploaded_file_name']}")

    if st.button("Process File"):
        if uploaded_file is None:
            st.warning("Please upload a file before processing.")
            st.stop()

        status_message = st.empty()
        result_area = st.empty()

        status_message.info("Waiting for backend response...")
        store_results, exception_rows = process_file(uploaded_file, subject, threshold)
        status_message.empty()
        if not store_results and not exception_rows:
            st.info("No valid rows found in file.")
        if store_results:
            st.success(f"{len(store_results)} question(s) processed and stored.")
            result_area.dataframe(build_results_dataframe(store_results), use_container_width=True)
        if exception_rows:
            st.warning(f"{len(exception_rows)} question(s) flagged and not stored.")
            st.dataframe(
                pd.DataFrame(exception_rows)[["qid", "raw_question", "enriched_text", "violations"]],
                use_container_width=True,
            )

    if st.session_state.get("last_results"):
        if st.session_state.get("last_status") == "success":
            status_message.success("Showing last successful result from {}.".format(st.session_state.get("last_results_source")))
            result_area.dataframe(pd.DataFrame(st.session_state["last_results"]), use_container_width=True)
        elif st.session_state.get("last_status") == "error":
            status_message.error("Last operation failed: {}".format(st.session_state.get("last_error")))


def show() -> None:
    subject = st.session_state["subject"]
    threshold = st.session_state["threshold"]
    
    tab_upload, tab_manual = st.tabs(["File Upload", "Manual Entry"])
    with tab_upload:
        render_file_upload_section(subject, threshold)
    with tab_manual:
        render_manual_entry_section(subject, threshold)


show()
