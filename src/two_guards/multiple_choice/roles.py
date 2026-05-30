"""LLM role functions for Plan B (Multiple Choice)."""

from two_guards.core.llm import complete_json
from two_guards.multiple_choice.prompts import (
    GENERATOR_SYSTEM,
    GENERATOR_USER,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
)


def run_generator(
    document_text: str,
    hallucination_type: str,
    model: str,
    reasoning_budget: int,
    max_attempts: int = 3,
) -> dict:
    """Generate one plausible but incorrect answer option for a document.

    Uses extended thinking so the fabrication reasoning is captured.

    Args:
        document_text: The source legal document.
        hallucination_type: The type of error to introduce (e.g. "date_error").
            Determines which prompt template is used.
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with key: fabricated_option and reasoning_tokens.
    """
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": GENERATOR_USER.format(
            document_text=document_text,
            hallucination_type=hallucination_type,
        )},
    ]
    return complete_json(
        model=model,
        messages=messages,
        max_attempts=max_attempts,
        thinking=True,
        reasoning_budget=reasoning_budget,
    )


def run_verifier(
    document_text: str,
    options: list[str],
    model: str,
    reasoning_budget: int,
    max_attempts: int = 3,
) -> dict:
    """Select the correct answer from a list of options based on the document.

    Args:
        document_text: The source legal document.
        options: List of statement texts (including the true statement).
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: choice_index (0-based), reasoning, reasoning_tokens.
    """
    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": VERIFIER_USER.format(
            document_text=document_text,
            options_text=options_text,
        )},
    ]
    return complete_json(
        model=model,
        messages=messages,
        max_attempts=max_attempts,
        thinking=True,
        reasoning_budget=reasoning_budget,
    )
