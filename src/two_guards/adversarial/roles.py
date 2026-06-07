"""LLM role functions for Plan A (Adversarial Conversation)."""

import json

from two_guards.adversarial.prompts import (
    LIAR_SYSTEM,
    LIAR_USER_FIRST,
    LIAR_USER_CONTINUE,
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
        hallucination_types: dict[str, dict[str, str]],
        prior_verifier_response: str | None,
        lies_so_far: list[str],
        model: str,
        reasoning_budget: int,
        max_attempts: int = 3,
) -> dict:
    """Generate a response to a legal document, optionally introducing a falsehood.
    Uses extended thinking so the fabrication reasoning is captured.

    Args:
        document_text: The source legal document.
        truth_flag: If False, the model is instructed to introduce one subtle lie.
        hallucination_type: The type of hallucination the liar is asked to generate.
        hallucination_types: All hallucination types, with a description and an example.
        prior_verifier_response: The verifier's last response text, None on the first turn.
        lies_so_far: List of claims already made in the conversation, used to prevent repetition.
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: response, truth_flag, false_claim (or None),
        hallucination_type, reasoning_tokens.
    """

    hallucination_info = "" if truth_flag else LIAR_USER_HALLUCINATION_INFO.format(
        hallucination_type=hallucination_type.lower(),
    )

    if prior_verifier_response is None:
        user_content = LIAR_USER_FIRST.format(
            document_text=document_text,
            truth_flag=str(truth_flag).lower(),
            hallucination_info=hallucination_info,
        )
    else:
        user_content = LIAR_USER_CONTINUE.format(
            verifier_response=prior_verifier_response,
            truth_flag=str(truth_flag).lower(),
            hallucination_info=hallucination_info,
            lies_so_far=json.dumps(lies_so_far, ensure_ascii=False),
        )

    messages = [
        {"role": "system", "content": LIAR_SYSTEM.format(
            hallucination_types='/'.join(hallucination_types.keys()).lower(),
            hallucination_types_info=str(hallucination_types)
        )},
        {"role": "user", "content": user_content},
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
        liar_response: str,
        model: str,
        reasoning_budget: int,
        max_attempts: int = 3,
) -> dict:
    """Fact-check a response against the source document.
    Uses extended thinking so the verification reasoning is captured.

    Args:
        document_text: The source legal document.
        liar_response: The response to verify.
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: attempted_correction, targeted_claim (or None),
        response, reasoning_tokens.
    """
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": VERIFIER_USER.format(
            document_text=document_text,
            liar_response=liar_response,
        )},
    ]

    return complete_json(
        model=model,
        messages=messages,
        max_attempts=max_attempts,
        thinking=True,
        reasoning_budget=reasoning_budget,
    )


def run_judge(
        document_text: str,
        turn_to_assess: dict,
        model: str,
        reasoning_budget: int = 8000,
        max_attempts: int = 3,
) -> dict:
    """Assess the most recent liar/verifier turn to determine whether
    1. the verifier correctly identified the liar's false claim.
    2. the liar did not introduce more hallucinations than what it claimed

    Args:
        document_text: The full source legal document.
        turn_to_assess:The current turn of the conversation, to be assessed by the judge.
        model: litellm model string.
        reasoning_budget: Token budget for the thinking trace.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with keys: undeclared_hallucinations, did_liar_hallucinate, verifier_correctly_identified,
        reasoning, reasoning_tokens.
    """

    turn_info_for_judge = {
        "liar_claim": turn_to_assess["liar"]["claim"],
        "liar_truth_flag": turn_to_assess["liar"]["truth_flag"],
        "liar_false_claim": turn_to_assess["liar"]["false_claim"],
        "verifier_attempted_correction": turn_to_assess["verifier"]["attempted_correction"],
        "verifier_targeted_claim": turn_to_assess["verifier"]["targeted_claim"],
        "verifier_response": turn_to_assess["verifier"]["response"],
        "was_lie_uncaught_this_turn": turn_to_assess["was_lie_uncaught_this_turn"],
    }

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            document_text=document_text,
            turn_json=json.dumps(turn_info_for_judge),
        )},
    ]
    return complete_json(
        model=model,
        messages=messages,
        max_attempts=max_attempts,
        thinking=True,
        reasoning_budget=reasoning_budget,
    )
