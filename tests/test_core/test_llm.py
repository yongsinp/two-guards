from unittest.mock import MagicMock, patch

import pytest

from two_guards.core.llm import complete, LLMJsonError, complete_json


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
    message.reasoning_content = None
    message.thinking = None
    message.thinking_content = None
    message.thinking_blocks = None
    mock.choices = [choice]
    choice.message = message
    mock._hidden_params = {}
    if thinking_content:
        mock._hidden_params["reasoning_content"] = thinking_content
    return mock


def _mock_response_message_reasoning(content: str, reasoning_content: str):
    mock = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    message.reasoning_content = reasoning_content
    message.thinking_blocks = None
    mock.choices = [choice]
    choice.message = message
    mock._hidden_params = {}
    return mock


def _mock_response_thinking_blocks(content: str, blocks: list[str]):
    mock = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    message.reasoning_content = None
    message.thinking = None
    message.thinking_content = None
    message.thinking_blocks = [MagicMock(thinking=b) for b in blocks]
    mock.choices = [choice]
    choice.message = message
    mock._hidden_params = {}
    return mock


def test_complete_basic():
    with patch("litellm.completion") as mock_completion:
        content = "Hello world"
        mock_completion.return_value = _mock_response(content)

        result = complete(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert result["content"] == content
        assert result["reasoning_tokens"] is None
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
            reasoning_budget=8000,
        )

        assert result["content"] == content
        assert result["reasoning_tokens"] == reasoning
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8000}


def test_complete_extracts_reasoning_from_message_field():
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_response_message_reasoning(
            content="Answer",
            reasoning_content="Reasoning from message field",
        )

        result = complete(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Question"}],
            thinking=True,
            reasoning_budget=8000,
        )

        assert result["reasoning_tokens"] == "Reasoning from message field"


def test_complete_extracts_reasoning_from_thinking_blocks():
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_response_thinking_blocks(
            content="Answer",
            blocks=["step one", "step two"],
        )

        result = complete(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Question"}],
            thinking=True,
            reasoning_budget=8000,
        )

        assert result["reasoning_tokens"] == "step one\nstep two"


@patch("two_guards.core.llm.complete")
def test_complete_json_success_first_attempt(mock_complete):
    mock_complete.return_value = {"content": '{"key": "value"}', "reasoning_tokens": None}

    response = complete_json(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_attempts=3,
    )

    assert response == {"key": "value", "reasoning_tokens": None}
    assert mock_complete.call_count == 1


@patch("two_guards.core.llm.complete")
def test_complete_json_retries_on_invalid_json(mock_complete):
    mock_complete.side_effect = [
        {"content": "not json", "reasoning_tokens": None},
        {"content": '{"key": "value"}', "reasoning_tokens": None},
    ]

    response = complete_json(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_attempts=3,
    )

    assert response == {"key": "value", "reasoning_tokens": None}
    assert mock_complete.call_count == 2


@patch("two_guards.core.llm.complete")
def test_complete_json_raises_after_exhausting_attempts(mock_complete):
    mock_complete.return_value = {"content": "not json at all", "reasoning_tokens": None}

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
    mock_complete.return_value = {"content": '{"key": "value"}', "reasoning_tokens": "some reasoning"}

    response = complete_json(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        max_attempts=3,
        thinking=True,
        reasoning_budget=1000,
    )

    assert response == {"key": "value", "reasoning_tokens": "some reasoning"}
