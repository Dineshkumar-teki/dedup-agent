# SQL Dedup Agent

LLM-powered question enrichment and vector-based deduplication for practice question pools. The app enriches raw questions into intent-focused sentences, embeds them, and checks or stores them in a Chroma vector database.

## Features

- **Store Pool** — Enrich and store questions with duplicate detection and validation
- **Check Pool** — Check questions against the pool without writing
- **CLI** — Batch enrich or store from CSV/XLSX files
- **Subjects** — SQL, Python, JavaScript, React, DSA (extensible via prompts)
- **Production-ready** — Configurable via env vars, optional auth, Docker, retries, upload limits

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)
- OpenRouter API key

### Setup

```bash
git clone <repo-url>
cd sql-dedup-agent
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
uv sync
```

### Run the web UI

```bash
uv run streamlit run app.py
```

Open http://localhost:8501

### Run the CLI

```bash
# Enrich only
uv run python main.py -i questions.csv --agent enricher --subject sql

# Enrich and store in Chroma
uv run python main.py -i questions.csv --agent rag --store --subject sql

# Single question
uv run python main.py --qid Q1 --question "What is a JOIN?" --subject sql
```

## Configuration

All settings are loaded from environment variables (see `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | **Required.** OpenRouter API key |
| `APP_USERNAME` / `APP_PASSWORD` | — | Enable login gate when both are set |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Vector DB storage path |
| `DUPLICATE_DISTANCE_THRESHOLD` | `0.25` | Cosine distance threshold (lower = stricter) |
| `ENRICH_CONCURRENCY` | `10` | Parallel LLM calls for batch uploads |
| `MAX_UPLOAD_ROWS` | `500` | Max rows per file upload |
| `MAX_UPLOAD_BYTES` | `5242880` | Max upload file size (5 MB) |

## Docker

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY and APP_USERNAME/APP_PASSWORD
docker compose up --build
```

Chroma data is persisted in a Docker volume (`chroma_data`).

## Deploy on Render

This app can run on [Render](https://render.com/) as a Docker web service. Chroma stays on a persistent disk.

1. Push this repo to GitHub.
2. In Render, click **New → Blueprint** and select the repo (it reads `render.yaml`).
3. Set these environment variables (do not commit them):
   - `OPENROUTER_API_KEY`
   - `APP_USERNAME`
   - `APP_PASSWORD`
4. Deploy. The app will be at `https://<service-name>.onrender.com`.

Use a **Starter** (paid) plan so the Chroma disk survives restarts. A free instance will lose the question pool when the service sleeps or restarts.

**Vercel is not supported.** Vercel is serverless and has no persistent disk, so Streamlit + local Chroma cannot run there.

## Deploy on Streamlit Community Cloud

This app is compatible with [Streamlit Community Cloud](https://share.streamlit.io/). Your repo already has `uv.lock`, which Streamlit uses to install dependencies.

### Steps

1. **Push to GitHub** — Streamlit Cloud deploys from a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **Create app**.
3. Select your repo, set **Main file path** to `app.py`, and choose **Python 3.12**.
4. Open **Advanced settings → Secrets** and paste the contents of `.streamlit/secrets.toml.example`, then fill in real values:

```toml
OPENROUTER_API_KEY = "sk-or-..."
APP_USERNAME = "admin"
APP_PASSWORD = "your-secure-password"
ENRICH_CONCURRENCY = "5"
```

5. Click **Deploy**.

### Important limitations on Streamlit Cloud

| Topic | Notes |
|-------|-------|
| **Chroma persistence** | The filesystem is ephemeral. Stored questions are lost when the app reboots or redeploys. Fine for demos; use Docker or a VPS for a persistent pool. |
| **Memory** | Chroma + LangChain can be heavy. Start with `ENRICH_CONCURRENCY = "3"` if the app crashes on large uploads. |
| **Secrets** | Never commit `.streamlit/secrets.toml`. Use Streamlit Cloud's Secrets UI instead. |
| **Auth** | Set `APP_USERNAME` and `APP_PASSWORD` in secrets — the app is public without them. |

For local dev with secrets, copy the example file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your keys
uv run streamlit run app.py
```

## Input file format

CSV or XLSX with columns for question ID and question text. Accepted aliases:

- **QID:** `qid`, `question id`, `question_id`, `id`
- **Question:** `question`, `raw_question`, `text`, `sql_question`

## Architecture

```text
User upload / manual entry
        │
        ▼
  Agent 1 (LLM enrich)  ──►  OpenRouter / Gemini
        │
        ▼
  Agent 2 (Chroma RAG)  ──►  OpenRouter embeddings + local Chroma DB
        │
        ▼
  Store / duplicate / new result
```

## Development

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run pytest -q
```

## Production checklist

- [ ] Set `OPENROUTER_API_KEY`
- [ ] Set `APP_USERNAME` and `APP_PASSWORD`
- [ ] Mount `CHROMA_PERSIST_DIR` to persistent storage
- [ ] Place the app behind HTTPS (reverse proxy)
- [ ] Do **not** commit `chroma_db/` — it is gitignored by default
- [ ] Monitor OpenRouter usage and tune `ENRICH_CONCURRENCY`

## License

Private / internal use.
