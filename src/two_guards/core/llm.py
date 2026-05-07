"""LLM wrapper around litellm with extended thinking support."""

from dataclasses import dataclass

import litellm


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider.

    Attributes:
        content: The text response from the model.
        reasoning: Reasoning tokens, or None if the model
            does not support them or reasoning was not requested.
    """

    content: str
    reasoning: str | None = None


def complete(
    model: str,
    messages: list[dict],
    thinking: bool = False,
    budget_tokens: int = 8000,
) -> LLMResponse:
    """Call an LLM via litellm and return a normalized response.

    Args:
        model: litellm model string, e.g. "anthropic/claude-sonnet-4-6".
        messages: OpenAI-style message list (role/content dicts).
        thinking: Whether to enable extended thinking / reasoning tokens.
        budget_tokens: Token budget for the thinking block when thinking=True.

    Returns:
        LLMResponse with the model's text output and optional reasoning trace.
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
    }

    if thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

    response = litellm.completion(**kwargs)

    content = response.choices[0].message.content or ""
    reasoning = response._hidden_params.get("reasoning_content")

    return LLMResponse(content=content, reasoning=reasoning)
