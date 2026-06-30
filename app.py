from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from agent2.storage import ChromaRAGStore, StoreResult

try:
    from agent1.agent import enrich_question
    from agent2.rag import process_file as process_rag_file
    from agent2.storage import ChromaRAGStore, StoreResult
    agent_import_error = None
except ImportError as exc:
    enrich_question = None
    process_rag_file = None
    ChromaRAGStore = None
    StoreResult = None
    agent_import_error = exc


def configure_page() -> None:
    st.set_page_config(page_title="Question Duplicate Checker", layout="centered")


def init_session_state() -> None:
    if "uploaded_file_bytes" not in st.session_state:
        st.session_state["uploaded_file_bytes"] = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "last_results_source" not in st.session_state:
        st.session_state["last_results_source"] = None
    if "last_status" not in st.session_state:
        st.session_state["last_status"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "pending_duplicate" not in st.session_state:
        st.session_state["pending_duplicate"] = None


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def load_uploaded_file_bytes(file_bytes: bytes, name: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    buffer.name = name
    if name.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


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


def get_rag_store() -> Any:
    if agent_import_error is not None:
        st.error(
            "Backend imports failed: {}. Install dependencies and run again.".format(agent_import_error)
        )
        return None

    if "rag_store" not in st.session_state:
        st.session_state["rag_store"] = ChromaRAGStore(persist_dir="chroma_db")
    return st.session_state["rag_store"]


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

        similarity_value = f"{similarity:.1f}%" if isinstance(similarity, (float, int)) else None
        results.append(
            {
                "Question ID": qid,
                "Status": status,
                "Original QID": original_qid,
                "Similarity": similarity_value,
                "Message": message,
            }
        )
    return pd.DataFrame(results)


def process_file(uploaded_file):
    upload_source, file_name = get_uploaded_file_source(uploaded_file)
    if upload_source is None:
        st.warning("Please select a file before clicking Process File.")
        return None

    rag_store = get_rag_store()
    if rag_store is None:
        return None

    temp_path = None
    try:
        if hasattr(upload_source, "getbuffer"):
            temp_path = save_uploaded_file_to_temp(upload_source)
        else:
            temp_path = save_uploaded_file_to_temp(upload_source)
        results = process_rag_file(str(temp_path), rag_store)
        st.session_state["last_results"] = [
            {
                "qid": item.qid,
                "status": item.status,
                "original_qid": item.original_qid,
                "similarity": getattr(item, "similarity", None),
                "message": item.message,
            }
            for item in results
        ]
        st.session_state["last_results_source"] = file_name
        st.session_state["last_status"] = "success"
        st.session_state["last_error"] = None
        return build_results_dataframe(results)
    except Exception as exc:
        st.error(f"Failed to process file: {exc}")
        st.session_state["last_status"] = "error"
        st.session_state["last_error"] = str(exc)
        return None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def process_manual_entry(question_id: str, question_text: str, allow_duplicate: bool = False):
    if not question_id.strip():
        st.warning("Question ID is empty")
        return None
    if not question_text.strip():
        st.warning("Question text is empty")
        return None

    rag_store = get_rag_store()
    if rag_store is None:
        return None

    try:
        enriched = enrich_question(question_id.strip(), question_text.strip())
        store_result = rag_store.add_or_flag_duplicate(
            question_id.strip(),
            question_text.strip(),
            enriched.enriched_text,
            allow_duplicate=allow_duplicate,
        )
        return store_result
    except Exception as exc:
        st.error(f"Failed to process question: {exc}")
        return None


def render_header() -> None:
    st.title("Question Duplicate Checker")
    st.write(
        "Upload a file or enter a question manually to enrich, deduplicate, and store SQL questions."
    )


def render_file_upload_section() -> None:
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
        status_message.info("Waiting for backend response...")
        df_results = process_file(uploaded_file)
        if df_results is not None:
            status_message.success("File processed successfully.")
            result_area.dataframe(df_results)
        else:
            status_message.error("Failed to process the file. See the message above.")

    if st.session_state.get("last_results"):
        if st.session_state.get("last_status") == "success":
            status_message.success("Showing last successful result from {}.".format(st.session_state.get("last_results_source")))
            result_area.dataframe(pd.DataFrame(st.session_state["last_results"]))
        elif st.session_state.get("last_status") == "error":
            status_message.error("Last operation failed: {}".format(st.session_state.get("last_error")))


def render_manual_entry_section() -> None:
    st.header("Manual Question Entry")
    question_id = st.text_input("Question ID")
    question_text = st.text_area("Question")

    status_message = st.empty()
    result_area = st.empty()

    if st.button("Check Question"):
        status_message.info("Checking question...")
        result = process_manual_entry(question_id, question_text)
        if result is None:
            status_message.error("Unable to check the question.")
            return

        if result.status == "duplicate":
            similarity_text = (
                f" ({result.similarity:.1f}%)" if isinstance(result.similarity, (float, int)) else ""
            )
            st.warning(f"Similar question found{similarity_text}. Click Keep to store it.")
            st.session_state["pending_duplicate"] = {
                "qid": question_id.strip(),
                "question_text": question_text.strip(),
            }
        else:
            st.session_state["pending_duplicate"] = None
            status_message.success(result.message or "Question processed.")

        result_area.dataframe(build_results_dataframe([result]))

    pending = st.session_state.get("pending_duplicate")
    if pending is not None:
        if st.button("Keep duplicate"):
            status_message.info("Saving duplicate...")
            result = process_manual_entry(
                pending["qid"],
                pending["question_text"],
                allow_duplicate=True,
            )
            st.session_state["pending_duplicate"] = None
            if result is not None:
                status_message.success(result.message or "Duplicate saved.")
                result_area.dataframe(build_results_dataframe([result]))
            else:
                status_message.error("Unable to save duplicate.")


def main() -> None:
    configure_page()
    init_session_state()

    with st.container():
        render_header()

    st.divider()

    with st.container():
        render_file_upload_section()

    st.divider()

    with st.container():
        render_manual_entry_section()


if __name__ == "__main__":
    main()
