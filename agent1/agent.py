from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from common.io import load_questions_from_file, write_enriched_questions
from common.prompts import ENRICHER_PROMPT

load_dotenv()


class EnrichedQuestion(BaseModel):
    qid: str = Field(description="The qid passed in, unchanged.")
    enriched_text: str = Field(
        description=(
            "One dense sentence capturing the question's logical intent, "
            "written per the enrichment rules in the system prompt."
        )
    )


def build_enricher() -> ChatGoogleGenerativeAI:
    """Create the AI agent configured to return the exact EnrichedQuestion shape."""
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    return model.with_structured_output(EnrichedQuestion)


def enrich_question(qid: str, raw_question: str) -> EnrichedQuestion:
    """Return an enriched result for the given qid/raw_question pair."""
    enricher = build_enricher()
    user_message = f"qid: {qid}\nraw_question: {raw_question}"
    return enricher.invoke([
        ("system", ENRICHER_PROMPT.system),
        ("human", user_message),
    ])


def process_file(input_file: str, output_file: str | None = None):
    rows = load_questions_from_file(input_file)
    if not rows:
        raise ValueError("No valid rows found in input file.")

    enriched = [enrich_question(qid, raw_question) for qid, raw_question in rows]
    write_enriched_questions([(item.qid, item.enriched_text) for item in enriched], output_file)
    return enriched
