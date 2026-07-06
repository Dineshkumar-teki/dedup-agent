import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from common.io import load_questions_from_file, write_enriched_questions
from common.prompts import get_prompt

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)


class EnrichedQuestion(BaseModel):
    qid: str = Field(description="The qid passed in, unchanged.")
    enriched_text: str = Field(
        description=(
            "One dense sentence capturing the question's logical intent, "
            "written per the enrichment rules in the system prompt."
        )
    )


def build_enricher() -> ChatOpenAI:
    """Create the AI agent configured to return the exact EnrichedQuestion shape."""
    load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the project .env file or export it before launching the app."
        )

    model = ChatOpenAI(
        model="google/gemini-3-flash-preview",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return model.with_structured_output(EnrichedQuestion)


def enrich_question(qid: str, raw_question: str, subject: str = "sql") -> EnrichedQuestion:
    """Return an enriched result for the given qid/raw_question pair."""
    enricher = build_enricher()
    user_message = f"qid: {qid}\nraw_question: {raw_question}"
    prompt = get_prompt(subject)
    return enricher.invoke([
        ("system", prompt.system),
        ("human", user_message),
    ])


def process_file(input_file: str, output_file: str | None = None, subject: str = "sql"):
    rows = load_questions_from_file(input_file)
    if not rows:
        raise ValueError("No valid rows found in input file.")

    enriched = [enrich_question(qid, raw_question, subject) for qid, raw_question in rows]
    write_enriched_questions([(item.qid, item.enriched_text) for item in enriched], output_file)
    return enriched
