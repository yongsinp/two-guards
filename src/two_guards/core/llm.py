"""LLM wrapper around litellm with extended thinking support."""

import json
import os

import litellm


class LLMJsonError(Exception):
    """Raised when the LLM fails to return valid JSON after all retry attempts."""


def complete(
    model: str,
    messages: list[dict],
    thinking: bool = False,
    reasoning_budget: int = 8000,
) -> dict:
    """Call an LLM via litellm and return a normalized response.

    Args:
        model: litellm model string, e.g. "anthropic/claude-sonnet-4-6".
        messages: OpenAI-style message list (role/content dicts).
        thinking: Whether to enable extended thinking / reasoning tokens.
        reasoning_budget: Token budget for the thinking block when thinking=True.

    Returns:
        Dict with keys: content, reasoning_tokens.
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
    }

    if thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": reasoning_budget}

    response = litellm.completion(**kwargs)

    content = response.choices[0].message.content or ""
    reasoning = response._hidden_params.get("reasoning_content")

    return {
        "content": content,
        "reasoning_tokens": reasoning,
    }


def complete_json(
    model: str,
    messages: list[dict],
    max_attempts: int = 3,
    thinking: bool = False,
    reasoning_budget: int = 8000,
) -> dict:
    """Call an LLM and parse the response as JSON, retrying on parse failures.

    Args:
        model: litellm model string.
        messages: OpenAI-style message list.
        max_attempts: Total number of attempts before raising LLMJsonError.
        thinking: Whether to enable extended thinking.
        reasoning_budget: Token budget for the thinking block when thinking=True.

    Returns:
        Parsed JSON payload with reasoning_tokens attached.

    Raises:
        LLMJsonError: If all attempts fail to produce valid JSON.
    """
    last_content = ""
    for _ in range(max_attempts):
        response = complete(
            model=model,
            messages=messages,
            thinking=thinking,
            reasoning_budget=reasoning_budget,
        )
        last_content = response["content"]
        try:
            last_content = _get_outermost_dict(last_content)
            parsed = json.loads(last_content)
            parsed.setdefault("reasoning_tokens", response.get("reasoning_tokens"))
            return parsed
        except ValueError:
            continue
    raise LLMJsonError(f"Failed to parse JSON after {max_attempts} attempts. Last response: {last_content!r}")

def _get_outermost_dict(text: str) -> str:
    """Extracts the outermost list from a string representation of a list.

    Args:
        text: The string to extract the list from.
    Returns:
        The extracted list as a string.
    Raises:
        ValueError: If failed to extract the list.
    """

    start = text.find('{')
    depth = 0

    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError("Failed to extract outermost dict from text")
