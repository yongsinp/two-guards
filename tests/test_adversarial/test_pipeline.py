from pathlib import Path
from unittest.mock import call, patch

from two_guards.adversarial.pipeline import run
from two_guards.adversarial.roles import run_liar, run_verifier, run_judge
from two_guards.core.config import Config, ModelConfig
from two_guards.core.loader import Document

DOC = Document(id="doc_001", text="The court ruled in 1983.", source_path="test.txt")

HALLUCINATION_TYPES = {
    "Temporal": {
        "description": "Time-sensitive errors and anachronisms.",
        "example": "The Build Back Better Act was signed under the Biden presidency in 2017.",
    }
}

def _make_config(tmp_path):
    return Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        reasoning_budget=8000,
    )

def test_run_liar_returns_parsed_output():
    liar_response = {
        "response": "The case was decided in 1985.",
        "truth_flag": False,
        "hallucination_type": "Temporal",
        "false_claim": "The case was decided in 1985 (actual: 1983).",
    }
    with patch("two_guards.adversarial.roles.complete_json") as mock:
        liar_response["reasoning_tokens"] = "I will change the year from 1983 to 1985."
        mock.return_value = liar_response
        result = run_liar(
            document_text="The landmark case was decided in 1983.",
            truth_flag=False,
            hallucination_type="Temporal",
            hallucination_types=HALLUCINATION_TYPES,
            prior_verifier_response=None,
            lies_so_far=[],
            model="anthropic/claude-sonnet-4-5",
            reasoning_budget=8000,
        )

    assert result["truth_flag"] is False
    assert result["hallucination_type"] == "Temporal"
    assert result["false_claim"] == "The case was decided in 1985 (actual: 1983)."
    assert result["reasoning_tokens"] == "I will change the year from 1983 to 1985."


def test_run_liar_passes_prior_response():
    liar_response = {
        "response": "The appeal was filed in 1990.",
        "truth_flag": False,
        "hallucination_type": "Temporal",
        "false_claim": "The appeal was filed in 1990 (actual: 1992).",
    }
    with patch("two_guards.adversarial.roles.complete_json") as mock:
        liar_response["reasoning_tokens"] = None
        mock.return_value = liar_response
        run_liar(
            document_text="The landmark case was decided in 1983.",
            truth_flag=False,
            hallucination_type="Temporal",
            hallucination_types=HALLUCINATION_TYPES,
            prior_verifier_response="The response appears accurate.",
            lies_so_far=["The case was decided in 1985 (actual: 1983)."],
            model="anthropic/claude-sonnet-4-5",
            reasoning_budget=8000,
        )

    user_message = mock.call_args[1]["messages"][1]["content"]
    assert "The response appears accurate." in user_message
    assert "1985 (actual: 1983)" in user_message


def test_run_verifier_returns_parsed_output():
    verifier_response = {
        "attempted_correction": True,
        "targeted_claim": "The year 1985 is incorrect.",
        "response": "The case was actually decided in 1983.",
    }
    with patch("two_guards.adversarial.roles.complete_json") as mock:
        verifier_response["reasoning_tokens"] = "The document says 1983, response says 1985."
        mock.return_value = verifier_response
        result = run_verifier(
            document_text="The case was decided in 1983.",
            liar_response="The case was decided in 1985.",
            model="anthropic/claude-sonnet-4-5",
            reasoning_budget=8000,
        )

    assert result["attempted_correction"] is True
    assert result["targeted_claim"] == "The year 1985 is incorrect."
    assert result["reasoning_tokens"] == "The document says 1983, response says 1985."


def test_run_judge_returns_parsed_output():
    judge_response = {
        "undeclared_hallucinations": [],
        "did_liar_hallucinate": False,
        "verifier_correctly_identified": True,
        "reasoning": "Both reference the same date error.",
    }
    turn_to_assess = {
        "liar": {
            "claim": "The case was decided in 1985.",
            "truth_flag": False,
            "false_claim": "The case was decided in 1985 (actual: 1983).",
        },
        "verifier": {
            "attempted_correction": True,
            "targeted_claim": "The year 1985 is incorrect.",
            "response": "The year should be 1983.",
        },
        "was_lie_uncaught_this_turn": False,
    }
    with patch("two_guards.adversarial.roles.complete_json") as mock:
        judge_response["reasoning_tokens"] = None
        mock.return_value = judge_response
        result = run_judge(
            document_text="The case was decided in 1983.",
            turn_to_assess=turn_to_assess,
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["verifier_correctly_identified"] is True
    assert result["did_liar_hallucinate"] is False
    assert result["undeclared_hallucinations"] == []


def test_pipeline_passed_record(tmp_path):
    liar_output = {
        "response": "The court ruled in 1985.",
        "truth_flag": False,
        "hallucination_type": "Temporal",
        "false_claim": "The court ruled in 1985 (actual: 1983).",
        "reasoning_tokens": "Changed year.",
    }
    verifier_output = {
        "attempted_correction": False,
        "targeted_claim": None,
        "response": "The response appears accurate.",
        "reasoning_tokens": "I found no issues.",
    }
    judge_output = {
        "verifier_correctly_identified": False,
        "reasoning": "Verifier did not attempt correction.",
        "reasoning_tokens": None,
    }

    with patch("two_guards.adversarial.pipeline.MAX_UNCAUGHT_LIES", 1), \
            patch("two_guards.adversarial.pipeline.run_liar", return_value=liar_output), \
            patch("two_guards.adversarial.pipeline.run_verifier", return_value=verifier_output), \
            patch("two_guards.adversarial.pipeline.run_judge", return_value=judge_output) as mock_judge, \
            patch("two_guards.adversarial.pipeline.write_record") as mock_write:
        run(config=_make_config(tmp_path), documents=[DOC])

    mock_judge.assert_not_called()

    assert mock_write.call_count == 2
    passed_call = next(c for c in mock_write.call_args_list if c[1]["passed"] is True)
    failed_call = next(c for c in mock_write.call_args_list if c[1]["passed"] is False)

    assert passed_call[1]["plan"] == "plan_a"
    assert len(passed_call[1]["record"]["turns"]) == 1
    assert passed_call[1]["record"]["turns"][0]["was_lie_uncaught_this_turn"] is True

    assert failed_call[1]["record"]["turns"] == []


def test_pipeline_failed_record_for_caught_lie(tmp_path):
    liar_output = {
        "response": "The court ruled in 1985.",
        "truth_flag": False,
        "hallucination_type": "Temporal",
        "false_claim": "The court ruled in 1985.",
        "reasoning_tokens": "Changed year.",
    }
    verifier_output = {
        "attempted_correction": True,
        "targeted_claim": "The year 1985 is wrong.",
        "response": "The year should be 1983.",
        "reasoning_tokens": "Source clearly says 1983.",
    }
    judge_output = {
        "undeclared_hallucinations": [],
        "did_liar_hallucinate": False,
        "verifier_correctly_identified": True,
        "reasoning": "Verifier correctly identified the date change.",
        "reasoning_tokens": "Compared false_claim and targeted_claim; they match",
    }

    with patch("two_guards.adversarial.pipeline.MAX_TOTAL_CLAIMS", 1), \
            patch("two_guards.adversarial.pipeline.run_liar", return_value=liar_output), \
            patch("two_guards.adversarial.pipeline.run_verifier", return_value=verifier_output), \
            patch("two_guards.adversarial.pipeline.run_judge", return_value=judge_output), \
            patch("two_guards.adversarial.pipeline.write_record") as mock_write:
        run(config=_make_config(tmp_path), documents=[DOC])

    assert mock_write.call_count == 2
    passed_call = next(c for c in mock_write.call_args_list if c[1]["passed"] is True)
    failed_call = next(c for c in mock_write.call_args_list if c[1]["passed"] is False)

    assert passed_call[1]["record"]["turns"] == []
    assert len(failed_call[1]["record"]["turns"]) == 1
    assert failed_call[1]["record"]["turns"][0]["was_lie_uncaught_this_turn"] is False


def test_pipeline_failed_record_for_truth_turn(tmp_path):
    liar_output = {
        "response": "The court ruled in 1983.",
        "truth_flag": True,
        "hallucination_type": None,
        "false_claim": None,
        "reasoning_tokens": None,
    }
    verifier_output = {
        "attempted_correction": False,
        "targeted_claim": None,
        "response": "The response is accurate.",
        "reasoning_tokens": None,
    }

    with patch("two_guards.adversarial.pipeline.MAX_TOTAL_CLAIMS", 1), \
            patch("two_guards.adversarial.pipeline.run_liar", return_value=liar_output), \
            patch("two_guards.adversarial.pipeline.run_verifier", return_value=verifier_output), \
            patch("two_guards.adversarial.pipeline.run_judge") as mock_judge, \
            patch("two_guards.adversarial.pipeline.write_record") as mock_write:
        run(config=_make_config(tmp_path), documents=[DOC])

    mock_judge.assert_not_called()

    assert mock_write.call_count == 2
    passed_call = next(c for c in mock_write.call_args_list if c[1]["passed"] is True)
    failed_call = next(c for c in mock_write.call_args_list if c[1]["passed"] is False)

    assert passed_call[1]["record"]["turns"] == []
    assert len(failed_call[1]["record"]["turns"]) == 1


def test_pipeline_judge_skipped_for_truth_turn(tmp_path):
    liar_output = {
        "response": "The court ruled in 1983.",
        "truth_flag": True,
        "hallucination_type": None,
        "false_claim": None,
        "reasoning_tokens": None,
    }
    verifier_output = {
        "attempted_correction": True,
        "targeted_claim": "Something.",
        "response": "I think this is wrong.",
        "reasoning_tokens": None,
    }

    with patch("two_guards.adversarial.pipeline.run_liar", return_value=liar_output), \
            patch("two_guards.adversarial.pipeline.run_verifier", return_value=verifier_output), \
            patch("two_guards.adversarial.pipeline.run_judge") as mock_judge, \
            patch("two_guards.adversarial.pipeline.write_record"):
        run(config=_make_config(tmp_path), documents=[DOC])

    mock_judge.assert_not_called()


test_run_liar_returns_parsed_output()
test_run_liar_passes_prior_response()
test_run_verifier_returns_parsed_output()
test_pipeline_passed_record(Path('.'))
test_pipeline_failed_record_for_caught_lie(Path('.'))
test_pipeline_failed_record_for_truth_turn(Path('.'))
test_pipeline_judge_skipped_for_truth_turn(Path('.'))
