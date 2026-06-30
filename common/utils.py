from typing import Any

QID_ALIASES = ["qid", "question id", "question_id", "id"]
QUESTION_ALIASES = ["question", "raw_question", "raw question", "text", "sql_question"]


def find_column(headers: dict[str, Any], aliases: list[str]) -> Any:
    """Return the value mapped to the first alias found in `headers`."""
    for alias in aliases:
        if alias in headers:
            return headers[alias]
    return None
