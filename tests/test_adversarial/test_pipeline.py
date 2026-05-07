import json
from unittest.mock import patch

from two_guards.core.llm import LLMResponse
from two_guards.core.loader import Document
from two_guards.core.config import Config, ModelConfig, ThinkingConfig
from two_guards.adversarial.roles import run_liar, run_verifier, run_judge
from two_guards.adversarial.pipeline import run


def test_run_liar_returns_parsed_output():
    liar_json = json.dumps({
        "response": "The case was decided in 1985.",
        "truth_flag": False,
        "false_claim": "The case was decided in 1985 (actual: 1983).",
    })
    with patch("two_guards.adversarial.roles.complete") as mock:
        mock.return_value = LLMResponse(
            content=liar_json,
            reasoning="I will change the year from 1983 to 1985.",
        )
        result = run_liar(
            document_text="The landmark case was decided in 1983.",
            truth_flag=False,
            model="anthropic/claude-sonnet-4-5",
            budget_tokens=8000,
        )

    assert result["truth_flag"] is False
    assert result["false_claim"] == "The case was decided in 1985 (actual: 1983)."
    assert result["reasoning"] == "I will change the year from 1983 to 1985."


def test_run_verifier_returns_parsed_output():
    verifier_json = json.dumps({
        "attempted_correction": True,
        "targeted_claim": "The year 1985 is incorrect.",
        "response": "The case was actually decided in 1983.",
    })
    with patch("two_guards.adversarial.roles.complete") as mock:
        mock.return_value = LLMResponse(
            content=verifier_json,
            reasoning="The document says 1983, response says 1985.",
        )
        result = run_verifier(
            document_text="The case was decided in 1983.",
            liar_response="The case was decided in 1985.",
            model="anthropic/claude-sonnet-4-5",
            budget_tokens=8000,
        )

    assert result["attempted_correction"] is True
    assert result["targeted_claim"] == "The year 1985 is incorrect."
    assert result["reasoning"] == "The document says 1983, response says 1985."


def test_run_judge_returns_parsed_output():
    judge_json = json.dumps({
        "verifier_correctly_identified": True,
        "reasoning": "Both reference the same date error.",
    })
    with patch("two_guards.adversarial.roles.complete") as mock:
        mock.return_value = LLMResponse(content=judge_json, reasoning=None)
        result = run_judge(
            false_claim="The case was decided in 1985 (actual: 1983).",
            targeted_claim="The year 1985 is incorrect.",
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["verifier_correctly_identified"] is True


def test_pipeline_passed_record(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        thinking=ThinkingConfig(budget_tokens=8000),
    )
    doc = Document(id="doc_001", text="The court ruled in 1983.", source_path="test.txt")

    liar_output = {
        "response": "The court ruled in 1985.",
        "truth_flag": False,
        "false_claim": "The court ruled in 1985 (actual: 1983).",
        "reasoning": "Changed year.",
    }
    verifier_output = {
        "attempted_correction": False,
        "targeted_claim": None,
        "response": "The response appears accurate.",
        "reasoning": "I found no issues.",
    }
    judge_output = {
        "verifier_correctly_identified": False,
        "reasoning": "Verifier did not attempt correction.",
    }

    with patch("two_guards.adversarial.pipeline.run_liar", return_value=liar_output), \
         patch("two_guards.adversarial.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.adversarial.pipeline.run_judge", return_value=judge_output), \
         patch("two_guards.adversarial.pipeline.write_record") as mock_write:

        run(config=config, documents=[doc])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["passed"] is True
        assert call_kwargs["plan"] == "plan_a"


def test_pipeline_failed_record(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        thinking=ThinkingConfig(budget_tokens=8000),
    )
    doc = Document(id="doc_001", text="The court ruled in 1983.", source_path="test.txt")

    liar_output = {
        "response": "The court ruled in 1985.",
        "truth_flag": False,
        "false_claim": "The court ruled in 1985.",
        "reasoning": "Changed year.",
    }
    verifier_output = {
        "attempted_correction": True,
        "targeted_claim": "The year 1985 is wrong.",
        "response": "The year should be 1983.",
        "reasoning": "Source clearly says 1983.",
    }
    judge_output = {
        "verifier_correctly_identified": True,
        "reasoning": "Verifier correctly identified the date change.",
    }

    with patch("two_guards.adversarial.pipeline.run_liar", return_value=liar_output), \
         patch("two_guards.adversarial.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.adversarial.pipeline.run_judge", return_value=judge_output), \
         patch("two_guards.adversarial.pipeline.write_record") as mock_write:

        run(config=config, documents=[doc])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["passed"] is False
