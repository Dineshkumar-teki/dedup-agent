"""LLM-based question enrichment via OpenRouter."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from common.config import get_settings
from common.io import load_questions_from_file, write_enriched_questions
from common.prompts import get_prompt

logger = logging.getLogger(__name__)


class EnrichedQuestion(BaseModel):
    qid: str = Field(description="The qid passed in, unchanged.")
    enriched_text: str = Field(
        description=(
            "One dense sentence capturing the question's logical intent, "
            "written per the enrichment rules in the system prompt."
        )
    )


@lru_cache(maxsize=1)
def build_enricher() -> ChatOpenAI:
    """Create a cached LLM client configured for structured EnrichedQuestion output."""
    settings = get_settings()
    settings.require_api_key()

    base_url = "https://api.openai.com/v1" if settings.api_provider == "openai" else "https://openrouter.ai/api/v1"
    model = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=base_url,
        api_key=settings.openrouter_api_key,
    )
    return model.with_structured_output(EnrichedQuestion)


def _invoke_enricher(qid: str, raw_question: str, subject: str) -> EnrichedQuestion:
    enricher = build_enricher()
    user_message = f"qid: {qid}\nraw_question: {raw_question}"
    prompt = get_prompt(subject)
    return enricher.invoke([
        ("system", prompt.system),
        ("human", user_message),
    ])


def enrich_question(qid: str, raw_question: str, subject: str = "sql") -> EnrichedQuestion:
    """Return an enriched result for the given qid/raw_question pair."""
    settings = get_settings()

    @retry(
        stop=stop_after_attempt(settings.api_retry_attempts),
        wait=wait_exponential(
            min=settings.api_retry_min_wait,
            max=settings.api_retry_max_wait,
        ),
        reraise=True,
    )
    def _call() -> EnrichedQuestion:
        return _invoke_enricher(qid, raw_question, subject)

    logger.debug("Enriching qid=%s subject=%s", qid, subject)
    return _call()


def process_file(input_file: str, output_file: str | None = None, subject: str = "sql"):
    rows = load_questions_from_file(input_file)
    if not rows:
        raise ValueError("No valid rows found in input file.")

    enriched = [enrich_question(qid, raw_question, subject) for qid, raw_question in rows]
    write_enriched_questions([(item.qid, item.enriched_text) for item in enriched], output_file)
    return enriched
