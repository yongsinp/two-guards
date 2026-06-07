"""LLM role functions for Plan B (Multiple Choice)."""

from two_guards.core.llm import complete_json
from two_guards.multiple_choice.prompts import (
    LIAR_SYSTEM,
    LIAR_USER,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
)


def run_liar(
    document_text: str,
    hallucination_type: str,
    all_hallucination_types: list[str],
    model: str,
    reasoning_budget: int,
    max_attempts: int = 3,
) -> dict:
    """Generate one plausible but incorrect answer option for a document.

    Uses extended thinking so the fabrication reasoning is captured.
    The source document is placed in the system prompt as a cached content
    block to reduce costs when the same document is processed multiple times.

    Args:
        document_text: The source legal document.
        hallucination_type: The type of error to introduce (e.g. "date_error").
            Determines which prompt template is used.
        all_hallucination_types: Full list of available lie types for context.
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with key: fabricated_option and reasoning_tokens.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", "text": LIAR_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                 },
                {
                    "type": "text",
                    "text": f"Source legal document:\n\n{document_text}",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "All possible lie types:\n\n" + "\n".join(
                        f"- {lie_type}" for lie_type in all_hallucination_types
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {"role": "user", "content": LIAR_USER.format(
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
    """Select all statements believed to be correct based on the document.

    All options are fabricated lies. The verifier is expected to return an
    empty list, but may select one or more indices if deceived.
    The source document is placed in the system prompt as a cached content
    block to reduce costs when the same document is processed multiple times.

    Args:
        document_text: The source legal document.
        options: List of fabricated statement texts.
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: choice_indices (list of 0-based ints), reasoning, reasoning_tokens.
    """
    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": VERIFIER_SYSTEM},
                {
                    "type": "text",
                    "text": f"Source legal document:\n\n{document_text}",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {"role": "user", "content": VERIFIER_USER.format(
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
