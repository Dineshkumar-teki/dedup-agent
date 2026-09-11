"""Frontend helper and Streamlit AppTest use cases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent2.storage import StoreResult
from app import build_results_dataframe, format_error
from common.ui import SAMPLE_CSV, STATUS_LABELS, summarize_results


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))
    monkeypatch.delenv("APP_USERNAME", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    from common.config import get_settings

    get_settings.cache_clear()
    return chroma_dir


def test_format_error_maps_known_failures() -> None:
    assert "API credentials" in format_error(RuntimeError("missing api_key"))
    assert "API credentials" in format_error(RuntimeError("invalid credential"))
    assert format_error(ValueError("QID column not found.")) == "QID column not found."
    assert "Network error" in format_error(RuntimeError("connection timeout"))
    assert "unexpected error" in format_error(RuntimeError("something else")).lower()


def test_build_results_dataframe_from_store_results() -> None:
    rows = [
        StoreResult("Q1", "stored", original_qid="Q1", similarity=100.0, message="Question stored."),
        StoreResult("Q2", "duplicate", original_qid="Q1", similarity=91.2, message="Similar question found."),
        StoreResult("Q3", "qid_conflict", original_qid="Q3", message="QID conflict: different question exists."),
        StoreResult("Q4", "new", original_qid="Q4"),
    ]
    df = build_results_dataframe(rows)
    assert list(df["Question ID"]) == ["Q1", "Q2", "Q3", "Q4"]
    assert list(df["Status"]) == [
        "Stored",
        "Duplicate",
        "ID conflict",
        "Unique",
    ]
    assert df.loc[0, "Similarity"] == "100.0%"
    assert df.loc[3, "Original QID"] == "Q4"
    assert df.loc[3, "Similarity"] == "—"


def test_build_results_dataframe_from_dicts() -> None:
    df = build_results_dataframe(
        [
            {
                "qid": "Q1",
                "status": "already_stored",
                "original_qid": None,
                "similarity": None,
                "message": "Question already stored.",
            }
        ]
    )
    assert df.loc[0, "Status"] == "Already stored"
    assert df.loc[0, "Original QID"] == "—"


def test_summarize_results_counts_statuses() -> None:
    counts = summarize_results(
        [
            StoreResult("Q1", "stored"),
            StoreResult("Q2", "duplicate"),
            StoreResult("Q3", "stored"),
            {"status": "new"},
        ]
    )
    assert counts == {"stored": 2, "duplicate": 1, "new": 1}


def test_status_labels_cover_store_and_check_outcomes() -> None:
    for status in (
        "stored",
        "duplicate",
        "new",
        "already_stored",
        "already_exists",
        "qid_conflict",
        "borderline_stored",
    ):
        assert status in STATUS_LABELS


def test_sample_csv_has_required_columns() -> None:
    df = pd.read_csv(pd.io.common.StringIO(SAMPLE_CSV))
    assert list(df.columns) == ["qid", "question"]
    assert len(df) == 2


def test_store_pool_renders_and_validates_empty_manual_entry(isolated_settings: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pages/store_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    assert not at.exception
    assert any("Store Pool" in str(item.value) for item in at.header)
    assert at.file_uploader
    button_labels = [item.label for item in at.button]
    assert "Process and store" in button_labels
    assert "Store question" in button_labels

    store_button = next(item for item in at.button if item.label == "Store question")
    store_button.click().run()
    assert any("Please enter a Question ID." in str(item.value) for item in at.warning)


def test_store_pool_validates_empty_question_text(isolated_settings: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pages/store_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    at.text_input[0].set_value("Q101").run()
    store_button = next(item for item in at.button if item.label == "Store question")
    store_button.click().run()
    assert any("Please enter the question text." in str(item.value) for item in at.warning)


def test_check_pool_renders_and_validates_empty_question(isolated_settings: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pages/check_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    assert not at.exception
    assert any("Check Pool" in str(item.value) for item in at.header)
    button_labels = [item.label for item in at.button]
    assert "Check for duplicates" in button_labels
    assert "Check question" in button_labels

    check_button = next(item for item in at.button if item.label == "Check question")
    check_button.click().run()
    assert any("Please enter the question text." in str(item.value) for item in at.warning)


def test_check_pool_defaults_missing_session_state(isolated_settings: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pages/check_pool.py", default_timeout=15)
    at.run()

    assert not at.exception
    assert any("Check Pool" in str(item.value) for item in at.header)


def test_app_login_gate_rejects_invalid_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.setenv("APP_PASSWORD", "secret")
    from common.config import get_settings

    get_settings.cache_clear()

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()

    assert not at.exception
    titles = [str(item.value) for item in at.title]
    assert any("Sign in" in title for title in titles)

    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("wrong-password")
    sign_in = next(item for item in at.button if item.label == "Sign in")
    sign_in.click().run()
    assert any("Invalid username or password." in str(item.value) for item in at.error)


def test_app_without_auth_renders_store_pool(isolated_settings: Path) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()

    assert not at.exception
    titles = [str(item.value) for item in at.title]
    assert any("Question Dedup" in title for title in titles)
    assert any("Store Pool" in str(item.value) for item in at.header)
    assert at.selectbox
    assert at.slider
    assert at.metric


def test_store_pool_manual_entry_success(isolated_settings: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "app.enrich_question",
        lambda qid, text, subject="sql": SimpleNamespace(qid=qid, enriched_text="intent a"),
    )
    monkeypatch.setattr(
        "app.get_rag_store",
        lambda subject, threshold: SimpleNamespace(
            add_or_flag_duplicate=lambda qid, raw, enriched, allow_duplicate=False: StoreResult(
                qid,
                "stored",
                original_qid=qid,
                message="Question stored.",
            )
        ),
    )

    at = AppTest.from_file("pages/store_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    at.text_input[0].set_value("Q101")
    at.text_area[0].set_value("What is a JOIN?")
    next(item for item in at.button if item.label == "Store question").click().run()

    assert any("Question stored successfully." in str(item.value) for item in at.success)
    assert not at.exception


def test_store_pool_manual_entry_duplicate_can_store_anyway(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "app.enrich_question",
        lambda qid, text, subject="sql": SimpleNamespace(qid=qid, enriched_text="intent a"),
    )

    def fake_add(qid, raw, enriched, allow_duplicate=False):
        if allow_duplicate:
            return StoreResult(qid, "stored", original_qid=qid, message="Duplicate stored.")
        return StoreResult(
            qid,
            "duplicate",
            original_qid="Q1",
            similarity=91.2,
            message="Similar question found.",
        )

    monkeypatch.setattr(
        "app.get_rag_store",
        lambda subject, threshold: SimpleNamespace(add_or_flag_duplicate=fake_add),
    )

    at = AppTest.from_file("pages/store_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    at.text_input[0].set_value("Q102")
    at.text_area[0].set_value("How do joins work?")
    next(item for item in at.button if item.label == "Store question").click().run()

    assert any("Similar question already in the pool" in str(item.value) for item in at.warning)
    button_labels = [item.label for item in at.button]
    assert "Store anyway" in button_labels
    assert "Cancel" in button_labels

    next(item for item in at.button if item.label == "Store anyway").click().run()
    assert any("Duplicate stored." in str(item.value) for item in at.success)


def test_store_pool_manual_entry_qid_conflict(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "app.enrich_question",
        lambda qid, text, subject="sql": SimpleNamespace(qid=qid, enriched_text="intent b"),
    )
    monkeypatch.setattr(
        "app.get_rag_store",
        lambda subject, threshold: SimpleNamespace(
            add_or_flag_duplicate=lambda qid, raw, enriched, allow_duplicate=False: StoreResult(
                qid,
                "qid_conflict",
                original_qid=qid,
                message="QID conflict: different question exists.",
            )
        ),
    )

    at = AppTest.from_file("pages/store_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    at.text_input[0].set_value("Q1")
    at.text_area[0].set_value("A different question")
    next(item for item in at.button if item.label == "Store question").click().run()
    assert any("already used for a different question" in str(item.value) for item in at.error)


def test_store_pool_manual_entry_already_stored(
    isolated_settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "app.enrich_question",
        lambda qid, text, subject="sql": SimpleNamespace(qid=qid, enriched_text="intent a"),
    )
    monkeypatch.setattr(
        "app.get_rag_store",
        lambda subject, threshold: SimpleNamespace(
            add_or_flag_duplicate=lambda qid, raw, enriched, allow_duplicate=False: StoreResult(
                qid,
                "already_stored",
                original_qid=qid,
                message="Question already stored.",
            )
        ),
    )

    at = AppTest.from_file("pages/store_pool.py", default_timeout=15)
    at.session_state["subject"] = "sql"
    at.session_state["threshold"] = 0.25
    at.run()

    at.text_input[0].set_value("Q1")
    at.text_area[0].set_value("What is a JOIN?")
    next(item for item in at.button if item.label == "Store question").click().run()
    assert any("already in the pool" in str(item.value) for item in at.info)
