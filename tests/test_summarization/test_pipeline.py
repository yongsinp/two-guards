import json
from unittest.mock import patch

from two_guards.core.llm import LLMResponse
from two_guards.core.loader import Document
from two_guards.core.config import Config, ModelConfig, ThinkingConfig
from two_guards.summarization.roles import (
    run_summarizer,
    run_tamperer,
    run_locator,
    run_judge,
)
from two_guards.summarization.pipeline import run


def test_run_summarizer():
    resp_json = json.dumps({"summary": "The court awarded $50,000 in damages."})
    with patch("two_guards.summarization.roles.complete") as mock:
        mock.return_value = LLMResponse(content=resp_json, reasoning=None)
        result = run_summarizer(
            document_text="Full document text...",
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["summary"] == "The court awarded $50,000 in damages."


def test_run_tamperer():
    resp_json = json.dumps({
        "tampered_summary": "The court awarded $75,000 in damages.",
        "introduced_errors": ["Changed damages from $50,000 to $75,000"],
    })
    with patch("two_guards.summarization.roles.complete") as mock:
        mock.return_value = LLMResponse(content=resp_json, reasoning=None)
        result = run_tamperer(
            summary="The court awarded $50,000 in damages.",
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["tampered_summary"] == "The court awarded $75,000 in damages."
    assert len(result["introduced_errors"]) == 1


def test_run_locator():
    resp_json = json.dumps({
        "located_errors": ["Damages amount is incorrect — should be $50,000 not $75,000"],
    })
    with patch("two_guards.summarization.roles.complete") as mock:
        mock.return_value = LLMResponse(content=resp_json, reasoning=None)
        result = run_locator(
            document_text="The court awarded $50,000 in damages.",
            tampered_summary="The court awarded $75,000 in damages.",
            model="anthropic/claude-sonnet-4-5",
        )

    assert len(result["located_errors"]) == 1


def test_run_judge():
    resp_json = json.dumps({
        "all_errors_found": True,
        "reasoning": "The fact-checker identified the damages amount error.",
    })
    with patch("two_guards.summarization.roles.complete") as mock:
        mock.return_value = LLMResponse(content=resp_json, reasoning=None)
        result = run_judge(
            introduced_errors=["Changed $50,000 to $75,000"],
            located_errors=["Damages should be $50,000 not $75,000"],
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["all_errors_found"] is True


def test_pipeline_passed_when_locator_misses_error(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        thinking=ThinkingConfig(budget_tokens=8000),
    )
    doc = Document(id="doc_001", text="The penalty is $10,000 under Section 5.", source_path="test.txt")

    summarizer_output = {"summary": "A $10,000 penalty applies under Section 5."}
    tamperer_output = {
        "tampered_summary": "A $15,000 penalty applies under Section 5.",
        "introduced_errors": ["Changed penalty from $10,000 to $15,000"],
    }
    locator_output = {"located_errors": []}
    judge_output = {
        "all_errors_found": False,
        "reasoning": "The fact-checker did not identify the damages error.",
    }

    with patch("two_guards.summarization.pipeline.run_summarizer", return_value=summarizer_output), \
         patch("two_guards.summarization.pipeline.run_tamperer", return_value=tamperer_output), \
         patch("two_guards.summarization.pipeline.run_locator", return_value=locator_output), \
         patch("two_guards.summarization.pipeline.run_judge", return_value=judge_output), \
         patch("two_guards.summarization.pipeline.write_record") as mock_write:

        run(config=config, documents=[doc])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["passed"] is True


def test_pipeline_failed_when_locator_finds_all(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        thinking=ThinkingConfig(budget_tokens=8000),
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    summarizer_output = {"summary": "A $10,000 penalty."}
    tamperer_output = {
        "tampered_summary": "A $15,000 penalty.",
        "introduced_errors": ["Changed $10,000 to $15,000"],
    }
    locator_output = {"located_errors": ["Amount should be $10,000 not $15,000"]}
    judge_output = {
        "all_errors_found": True,
        "reasoning": "Fact-checker correctly identified the amount error.",
    }

    with patch("two_guards.summarization.pipeline.run_summarizer", return_value=summarizer_output), \
         patch("two_guards.summarization.pipeline.run_tamperer", return_value=tamperer_output), \
         patch("two_guards.summarization.pipeline.run_locator", return_value=locator_output), \
         patch("two_guards.summarization.pipeline.run_judge", return_value=judge_output), \
         patch("two_guards.summarization.pipeline.write_record") as mock_write:

        run(config=config, documents=[doc])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["passed"] is False
