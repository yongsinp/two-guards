"""Plan C pipeline: Summarization (Summarizer → Tamperer → Locator → Judge)."""

from datetime import datetime, timezone

from two_guards.core.config import Config
from two_guards.core.loader import Document
from two_guards.core.writer import write_record
from two_guards.summarization.roles import (
    run_summarizer,
    run_tamperer,
    run_locator,
    run_judge,
)


def run(config: Config, documents: list[Document]) -> None:
    """Run the summarization pipeline over a list of documents.

    For each document:
    - A Summarizer produces an accurate summary.
    - A Tamperer introduces 1-3 subtle errors into the summary.
    - A Locator attempts to identify all errors by comparing to the source.
    - A Judge determines whether the Locator found every introduced error.

    Records where the Locator missed at least one error are written to
    ``passed/``; records where all errors were found go to ``failed/``.

    Args:
        config: Project configuration (model names, paths, thinking budget).
        documents: Source documents to process.
    """
    for doc in documents:
        summarizer_result = run_summarizer(
            document_text=doc.text,
            model=config.models.summarizer,
        )

        tamperer_result = run_tamperer(
            summary=summarizer_result["summary"],
            model=config.models.tamperer,
        )

        locator_result = run_locator(
            document_text=doc.text,
            tampered_summary=tamperer_result["tampered_summary"],
            model=config.models.locator,
        )

        judge_result = run_judge(
            introduced_errors=tamperer_result["introduced_errors"],
            located_errors=locator_result["located_errors"],
            model=config.models.judge,
        )

        passed = not judge_result["all_errors_found"]

        record = {
            "plan": "C",
            "document_id": doc.id,
            "original_summary": summarizer_result["summary"],
            "tampered_summary": tamperer_result["tampered_summary"],
            "introduced_errors": tamperer_result["introduced_errors"],
            "locator": {
                "located_errors": locator_result["located_errors"],
            },
            "judge": judge_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_record(
            output_dir=config.output_dir,
            plan="plan_c",
            passed=passed,
            record=record,
        )
