from unittest.mock import patch, MagicMock

from two_guards.core.llm import complete, LLMResponse


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
