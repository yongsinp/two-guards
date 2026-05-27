"""LLM role functions for Plan A (Adversarial Conversation)."""

from two_guards.adversarial.prompts import (
    LIAR_SYSTEM,
    LIAR_USER,
    LIAR_USER_HALLUCINATION_INFO,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
)
from two_guards.core.llm import complete_json


def run_liar(
        document_text: str,
        truth_flag: bool,
        hallucination_type: str | None,
        hallucination_types: dict[str, dict[str, str]] | None,
        model: str,
        budget_tokens: int,
        max_attempts: int = 3,
) -> dict:
    """Generate a response to a legal document, optionally introducing a falsehood.

    Uses extended thinking so the fabrication reasoning is captured.

    Args:
        document_text: The source legal document.
        truth_flag: If False, the model is instructed to introduce one subtle lie.
        hallucination_type: The type of hallucination the liar is asked to generate.
        hallucination_types: All hallucination types, with a description and an example.
        model: litellm model string.
        budget_tokens: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: response, truth_flag, false_claim (or None), reasoning.
    """

    hallucination_info = "" if truth_flag else LIAR_USER_HALLUCINATION_INFO.format(
        hallucination_type=hallucination_type.lower(),
    )

    messages = [
        {"role": "system", "content": LIAR_SYSTEM.format(
            hallucination_types='/'.join(hallucination_types.keys()).lower(),
            hallucination_types_info=str(hallucination_types)
        )},
        {"role": "user", "content": LIAR_USER.format(
            document_text=document_text,
            truth_flag=str(truth_flag).lower()
        ) + hallucination_info}
    ]

    return fetch_parsed_response(model, messages, budget_tokens, max_attempts)


def run_verifier(
        document_text: str,
        liar_response: str,
        model: str,
        budget_tokens: int,
        max_attempts: int = 3,
) -> dict:
    """Fact-check a response against the source document.

    Uses extended thinking so the verification reasoning is captured.

    Args:
        document_text: The source legal document.
        liar_response: The response to verify.
        model: litellm model string.
        budget_tokens: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: attempted_correction, targeted_claim (or None),
        response, reasoning.
    """
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": VERIFIER_USER.format(
            document_text=document_text,
            liar_response=liar_response,
        )},
    ]

    return fetch_parsed_response(model, messages, budget_tokens, max_attempts)


def fetch_parsed_response(model, messages, budget_tokens, max_attempts):
    parsed, reasoning = complete_json(
        model=model,
        messages=messages,
        max_attempts=max_attempts,
        thinking=True,
        budget_tokens=budget_tokens,
    )
    parsed["reasoning"] = reasoning
    return parsed


def run_judge(
        false_claim: str,
        targeted_claim: str,
        model: str,
        max_attempts: int = 3,
) -> dict:
    """Determine whether the verifier correctly identified the liar's false claim.

    Args:
        false_claim: The specific claim the liar fabricated.
        targeted_claim: The claim the verifier attempted to correct.
        model: litellm model string.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: verifier_correctly_identified, reasoning.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            false_claim=false_claim,
            targeted_claim=targeted_claim,
        )},
    ]
    parsed, _ = complete_json(model=model, messages=messages, max_attempts=max_attempts)
    return parsed
