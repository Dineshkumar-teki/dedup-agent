import csv
from pathlib import Path
from typing import Iterable

from .utils import find_column, QID_ALIASES, QUESTION_ALIASES


def load_questions_from_csv(path: str) -> list[tuple[str, str]]:
    """Load (qid, question) pairs from a CSV file.

    Raises ValueError if:
    - File has no headers.
    - QID or Question column cannot be found.
    Error message includes accepted aliases and actual headers found.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file must include headers with 'qid' and 'question'.")

        headers = {header.strip().lower(): header for header in reader.fieldnames}
        qid_key = find_column(headers, QID_ALIASES)
        question_key = find_column(headers, QUESTION_ALIASES)

        if qid_key is None:
            raise ValueError(
                f"QID column not found. "
                f"Accepted: {QID_ALIASES}. "
                f"Found: {list(headers.keys())}"
            )
        if question_key is None:
            raise ValueError(
                f"Question column not found. "
                f"Accepted: {QUESTION_ALIASES}. "
                f"Found: {list(headers.keys())}"
            )

        rows: list[tuple[str, str]] = []
        for row in reader:
            qid = (row.get(qid_key) or "").strip()
            question = (row.get(question_key) or "").strip()
            if qid and question:
                rows.append((qid, question))
        return rows


def load_questions_from_excel(path: str) -> list[tuple[str, str]]:
    """Load (qid, question) pairs from an Excel file (.xlsx/.xls).

    Raises ValueError if:
    - File is empty.
    - QID or Question column cannot be found.
    Error message includes accepted aliases and actual headers found.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "Excel support requires openpyxl. Install it with 'uv add openpyxl'."
        ) from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Excel file is empty.")

    header = [str(cell).strip().lower() if cell is not None else "" for cell in rows[0]]
    headers = {name: idx for idx, name in enumerate(header) if name}

    qid_idx = find_column(headers, QID_ALIASES)
    question_idx = find_column(headers, QUESTION_ALIASES)

    if qid_idx is None:
        raise ValueError(
            f"QID column not found. "
            f"Accepted: {QID_ALIASES}. "
            f"Found: {list(headers.keys())}"
        )
    if question_idx is None:
        raise ValueError(
            f"Question column not found. "
            f"Accepted: {QUESTION_ALIASES}. "
            f"Found: {list(headers.keys())}"
        )

    entries: list[tuple[str, str]] = []
    for row in rows[1:]:
        if not row:
            continue
        qid = str(row[qid_idx]).strip() if row[qid_idx] is not None else ""
        question = str(row[question_idx]).strip() if row[question_idx] is not None else ""
        if qid and question:
            entries.append((qid, question))
    return entries


def load_questions_from_file(input_path: str) -> list[tuple[str, str]]:
    """Dispatch to CSV or Excel loader based on file extension.

    Raises ValueError for unsupported file types.
    """
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        return load_questions_from_csv(str(path))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return load_questions_from_excel(str(path))
    raise ValueError(
        "Unsupported input file type. Use a CSV file or an Excel file with .xlsx/.xls extension."
    )


def write_enriched_questions(enriched: Iterable[tuple[str, str]], output_file: str | None = None) -> None:
    """Write (qid, enriched_text) pairs to a CSV file or stdout."""
    if output_file:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["qid", "enriched_text"])
            for qid, enriched_text in enriched:
                writer.writerow([qid, enriched_text])
        print(f"Wrote enriched results to {output_file}")
        return

    for qid, enriched_text in enriched:
        print("qid:", qid)
        print("enriched_text:", enriched_text)
        print()