"""LLM role functions for Plan B (Multiple Choice)."""

import json

from two_guards.core.llm import complete
from two_guards.multiple_choice.prompts import (
    GENERATOR_SYSTEM,
    GENERATOR_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
)


def run_generator(
    document_text: str,
    hallucination_type: str,
    model: str,
    budget_tokens: int,
) -> dict:
    """Generate one plausible but incorrect answer option for a document.

    Uses extended thinking so the fabrication reasoning is captured.

    Args:
        document_text: The source legal document.
        hallucination_type: The type of error to introduce (e.g. "date_error").
            Determines which prompt template is used.
        model: litellm model string.
        budget_tokens: Token budget for the thinking trace.

    Returns:
        Dict with keys: fabricated_option, question, reasoning.
    """
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": GENERATOR_USER.format(
            document_text=document_text,
            hallucination_type=hallucination_type,
        )},
    ]
    response = complete(model=model, messages=messages, thinking=True, budget_tokens=budget_tokens)
    parsed = json.loads(response.content)
    parsed["reasoning"] = response.reasoning
    return parsed


def run_judge(
    document_text: str,
    question: str,
    options: list[str],
    model: str,
) -> dict:
    """Select the correct answer from a list of options based on the document.

    Args:
        document_text: The source legal document.
        question: The multiple-choice question.
        options: List of answer option texts (including the true answer).
        model: litellm model string.

    Returns:
        Dict with keys: choice_index (0-based), reasoning.
    """
    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            document_text=document_text,
            question=question,
            options_text=options_text,
        )},
    ]
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)
