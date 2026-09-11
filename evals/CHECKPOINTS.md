# Checkpoints and expected output

Pipeline: **raw question / CSV → Agent 1 (enrich) → validation → Agent 2 (check/store)**

---

## CP0 — Input

| Check | Pass if |
|---|---|
| File or single question present | CSV/XLSX has `qid` + `question` (or aliases), or both ID and text are filled |
| Row is usable | `qid` and `question` are non-empty |
| Upload limits | ≤ 500 rows, ≤ 5 MB |

**Sample input (single)**

```text
qid:      Q1
question: Write a SQL query to get the second highest salary from the Employee table.
subject:  sql
```

**Sample input (CSV)**

```csv
qid,question
Q1,Write a SQL query to get the second highest salary from the Employee table.
Q2,What is the difference between INNER JOIN and LEFT JOIN?
Q3,Find the 2nd max salary of employees.
```

**Expected o/p if input fails**

```text
Please enter a Question ID.
Please enter the question text.
No valid rows found. Check your file format.
QID column not found.
File exceeds maximum upload size ...
```

---

## CP1 — Agent 1 (enrich)

| Check | Pass if |
|---|---|
| Output shape | `{qid, enriched_text}` |
| QID unchanged | `qid` matches input |
| Intent sentence | one prose sentence, operation named first |
| No leak | `qid` not inside `enriched_text` |
| No SQL/code | no `SELECT` / `def ` / `=>` as code |
| Ambiguous | ends with `[LOW_CONFIDENCE]` |

**Expected o/p (Q1)**

```text
qid: Q1
enriched_text: Ranking/window over employee salaries to select the second-highest salary value.
```

**Expected o/p (ambiguous: "Fix the query.")**

```text
qid: Q99
enriched_text: ... [LOW_CONFIDENCE]
```

---

## CP2 — Validation

| Input enriched_text | Expected |
|---|---|
| clean intent sentence | `[]` (no violations) → continue |
| contains qid | skip / flag: `qid leaked into enriched_text` |
| contains `SELECT` / `FROM` / `JOIN` | skip / flag: `code syntax detected` |
| contains `[LOW_CONFIDENCE]` | skip / flag: `LOW_CONFIDENCE flag present` |
| > 80 words | skip / flag: `enriched_text too long` |

**Expected o/p (batch UI)**

```text
N question(s) skipped due to errors.
```

---

## CP3 — Agent 2 store / check

Threshold: cosine **distance ≤ 0.25** ⇒ duplicate.

| Checkpoint | When | Expected status | Expected UI / message |
|---|---|---|---|
| New question | not in pool | `stored` (Store) / `new` (Check) | `Question stored successfully.` / `Unique — this question is not in the pool.` |
| Paraphrase duplicate | Q3 vs Q1 (2nd salary) | `duplicate` | `Similar question already in the pool (xx.x% similar).` + Store anyway / Cancel |
| Exact same QID + text | submit Q1 again | `already_stored` / `already_exists` | `This question is already in the pool.` |
| Same QID, different text | Q1 reused | `qid_conflict` | `This Question ID is already used for a different question.` |
| Near-miss | 2nd vs 3rd salary, INNER vs LEFT | `stored` / `new` | treated as unique |
| Borderline | distance ≈ 0.25 ± 0.03 | `borderline_stored` | stored, marked borderline |
| Store anyway | user confirms duplicate | `stored` | `Duplicate stored.` |

**Expected o/p for the sample CSV (Store Pool, empty pool)**

```text
Q1  Stored      —     Question stored.
Q2  Stored      —     Question stored.
Q3  Duplicate   Q1    Similar question found.    (not written unless Store anyway)
```

**Expected o/p (Check Pool, after Q1+Q2 stored)**

```text
Q1  Already in pool   Q1
Q2  Already in pool   Q2
Q3  Duplicate         Q1   similarity ~90%+
```

---

## CP4 — Result table (UI)

| Column | Expected |
|---|---|
| Question ID | input qid |
| Status | Stored / Duplicate / Unique / Already stored / Already in pool / ID conflict / Stored (borderline) |
| Original QID | match id, or `—` |
| Similarity | `91.2%` or `—` |
| Message | status message above |

Row color: green = stored/unique, red = duplicate/conflict, gray = already in pool.

---

## CP5 — Evals (offline)

```bash
uv run python -m evals
```

| Suite | Cases | Expected score |
|---|---|---|
| enrichment | 6 | 100.0 |
| validation | 6 | 100.0 |
| dedup_text | 12 pairs (P=1, R=1, F1=1, Acc=1) | 100.0 |
| **OVERALL** | 25% + 15% + 60% | **100.0 / 100** |
| CI floor | — | fail if overall < 85.0 |

**Expected scorecard**

```text
SQL Dedup Agent — Eval Scorecard
========================================
enrichment          100.0 / 100
validation          100.0 / 100
dedup_text          100.0 / 100
  P=1.00  R=1.00  F1=1.00  Acc=1.00
----------------------------------------
OVERALL             100.0 / 100
```
