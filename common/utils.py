# common/utils.py

import re
from typing import Any

QID_ALIASES = ["qid", "question id", "question_id", "id"]
QUESTION_ALIASES = ["question", "raw_question", "raw question", "text", "sql_question"]


def normalize(s: str) -> str:
    """Normalize a string for column alias matching.
    Strips whitespace, lowercases, and removes underscores, hyphens, and spaces.
    Example: 'Question_ID' -> 'questionid'
    """
    return re.sub(r"[\s_\-]", "", s.strip().lower())


def find_column(headers: dict[str, Any], aliases: list[str]) -> Any:
    """Return the value mapped to the first alias found in headers.
    Matching is case-insensitive and separator-agnostic (_, -, space).
    Returns None if no alias matches any header key.
    """
    normalized_aliases = {normalize(a) for a in aliases}
    for key in headers:
        if normalize(key) in normalized_aliases:
            return headers[key]
    return None