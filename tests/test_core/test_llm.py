from unittest.mock import MagicMock, patch

import pytest

from two_guards.core.llm import complete, LLMResponse, LLMJsonError, complete_json


def _mock_response(content: str, thinking_content: str | None = None):
    """Build a fake litellm response.

    Fields set:
        response.choices[0].message.content
        response._hidden_params["reasoning_content"] (optional, Claude only)
    """
    mock = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    mock.choices = [choice]
    choice.message = message
    mock._hidden_params = {}
    if thinking_content:
        mock._hidden_params["reasoning_content"] = thinking_content
    return mock


def test_complete_basic():
    with patch("litellm.completion") as mock_completion:
        content = "Hello world"
        mock_completion.return_value = _mock_response(content)

        result = complete(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == content
        assert result.reasoning is None
        mock_completion.assert_called_once()


def test_complete_with_thinking():
    with patch("litellm.completion") as mock_completion:
        content = "The answer is 42"
        reasoning = "Let me think about this step by step..."
        mock_completion.return_value = _mock_response(
            content=content,
            thinking_content=reasoning,
        )

        result = complete(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "What is the meaning?"}],
            thinking=True,
            budget_tokens=8000,
        )

        assert result.content == content
        assert result.reasoning == reasoning
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8000}


@patch("two_guards.core.llm.complete")
def test_complete_json_success_first_attempt(mock_complete):
    mock_complete.return_value = LLMResponse(content='{"key": "value"}')

    result, reasoning = complete_json(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_attempts=3,
    )

    assert result == {"key": "value"}
    assert reasoning is None
    assert mock_complete.call_count == 1


@patch("two_guards.core.llm.complete")
def test_complete_json_retries_on_invalid_json(mock_complete):
    mock_complete.side_effect = [
        LLMResponse(content="not json"),
        LLMResponse(content='{"key": "value"}'),
    ]

    result, reasoning = complete_json(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_attempts=3,
    )

    assert result == {"key": "value"}
    assert mock_complete.call_count == 2


@patch("two_guards.core.llm.complete")
def test_complete_json_raises_after_exhausting_attempts(mock_complete):
    mock_complete.return_value = LLMResponse(content="not json at all")

    with pytest.raises(LLMJsonError) as exc_info:
        complete_json(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            max_attempts=3,
        )

    assert mock_complete.call_count == 3
    assert "not json at all" in str(exc_info.value)


@patch("two_guards.core.llm.complete")
def test_complete_json_returns_reasoning(mock_complete):
    mock_complete.return_value = LLMResponse(content='{"key": "value"}', reasoning="some reasoning")

    result, reasoning = complete_json(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_attempts=3,
        thinking=True,
        budget_tokens=1000,
    )

    assert result == {"key": "value"}
    assert reasoning == "some reasoning"
