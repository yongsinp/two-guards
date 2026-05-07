"""LLM role functions for Plan A (Adversarial Conversation)."""

import json

from two_guards.core.llm import complete
from two_guards.adversarial.prompts import (
    LIAR_SYSTEM,
    LIAR_USER,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
)


def run_liar(
    document_text: str,
    truth_flag: bool,
    model: str,
    budget_tokens: int,
) -> dict:
    """Generate a response to a legal document, optionally introducing a falsehood.

    Uses extended thinking so the fabrication reasoning is captured.

    Args:
        document_text: The source legal document.
        truth_flag: If False, the model is instructed to introduce one subtle lie.
        model: litellm model string.
        budget_tokens: Token budget for the thinking trace.

    Returns:
        Dict with keys: response, truth_flag, false_claim (or None), reasoning.
    """
    messages = [
        {"role": "system", "content": LIAR_SYSTEM},
        {"role": "user", "content": LIAR_USER.format(
            document_text=document_text,
            truth_flag=str(truth_flag).lower(),
        )},
    ]
    response = complete(model=model, messages=messages, thinking=True, budget_tokens=budget_tokens)
    parsed = json.loads(response.content)
    parsed["reasoning"] = response.reasoning
    return parsed


def run_verifier(
    document_text: str,
    liar_response: str,
    model: str,
    budget_tokens: int,
) -> dict:
    """Fact-check a response against the source document.

    Uses extended thinking so the verification reasoning is captured.

    Args:
        document_text: The source legal document.
        liar_response: The response to verify.
        model: litellm model string.
        budget_tokens: Token budget for the thinking trace.

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
    response = complete(model=model, messages=messages, thinking=True, budget_tokens=budget_tokens)
    parsed = json.loads(response.content)
    parsed["reasoning"] = response.reasoning
    return parsed


def run_judge(
    false_claim: str,
    targeted_claim: str,
    model: str,
) -> dict:
    """Determine whether the verifier correctly identified the liar's false claim.

    Args:
        false_claim: The specific claim the liar fabricated.
        targeted_claim: The claim the verifier attempted to correct.
        model: litellm model string.

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
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)
