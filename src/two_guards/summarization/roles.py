"""LLM role functions for Plan C (Summarization)."""

import json

from two_guards.core.llm import complete
from two_guards.summarization.prompts import (
    SUMMARIZER_SYSTEM,
    SUMMARIZER_USER,
    TAMPERER_SYSTEM,
    TAMPERER_USER,
    LOCATOR_SYSTEM,
    LOCATOR_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
)


def run_summarizer(document_text: str, model: str) -> dict:
    """Generate an accurate summary of a legal document.

    Args:
        document_text: The source legal document.
        model: litellm model string.

    Returns:
        Dict with key: summary.
    """
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user", "content": SUMMARIZER_USER.format(document_text=document_text)},
    ]
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)


def run_tamperer(summary: str, model: str) -> dict:
    """Introduce 1-3 subtle factual errors into an accurate summary.

    Args:
        summary: The accurate summary to tamper with.
        model: litellm model string.

    Returns:
        Dict with keys: tampered_summary, introduced_errors (list of descriptions).
    """
    messages = [
        {"role": "system", "content": TAMPERER_SYSTEM},
        {"role": "user", "content": TAMPERER_USER.format(summary=summary)},
    ]
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)


def run_locator(document_text: str, tampered_summary: str, model: str) -> dict:
    """Identify factual errors in a summary by comparing it to the source document.

    Args:
        document_text: The original source document.
        tampered_summary: The summary that may contain errors.
        model: litellm model string.

    Returns:
        Dict with key: located_errors (list of descriptions, may be empty).
    """
    messages = [
        {"role": "system", "content": LOCATOR_SYSTEM},
        {"role": "user", "content": LOCATOR_USER.format(
            document_text=document_text,
            tampered_summary=tampered_summary,
        )},
    ]
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)


def run_judge(introduced_errors: list[str], located_errors: list[str], model: str) -> dict:
    """Determine whether the locator found all introduced errors.

    Args:
        introduced_errors: Descriptions of errors intentionally introduced.
        located_errors: Descriptions of errors the locator reported.
        model: litellm model string.

    Returns:
        Dict with keys: all_errors_found, reasoning.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            introduced_errors="\n".join(f"- {e}" for e in introduced_errors),
            located_errors="\n".join(f"- {e}" for e in located_errors),
        )},
    ]
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)
