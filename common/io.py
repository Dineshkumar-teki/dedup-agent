import csv
from pathlib import Path
from typing import Iterable

from .utils import find_column, QID_ALIASES, QUESTION_ALIASES


def load_questions_from_csv(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file must include headers with 'qid' and 'question'.")

        headers = {header.strip().lower(): header for header in reader.fieldnames}
        qid_key = find_column(headers, QID_ALIASES)
        question_key = find_column(headers, QUESTION_ALIASES)

        if qid_key is None or question_key is None:
            raise ValueError(
                "CSV file must contain columns named 'qid' and 'question' (or compatible aliases)."
            )

        rows: list[tuple[str, str]] = []
        for row in reader:
            qid = (row.get(qid_key) or "").strip()
            question = (row.get(question_key) or "").strip()
            if qid and question:
                rows.append((qid, question))
        return rows


def load_questions_from_excel(path: str) -> list[tuple[str, str]]:
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

    if qid_idx is None or question_idx is None:
        raise ValueError(
            "Excel file must contain columns named 'qid' and 'question' (or compatible aliases)."
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
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        return load_questions_from_csv(str(path))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return load_questions_from_excel(str(path))
    raise ValueError(
        "Unsupported input file type. Use a CSV file or an Excel file with .xlsx/.xls extension."
    )


def write_enriched_questions(enriched: Iterable[tuple[str, str]], output_file: str | None = None) -> None:
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
