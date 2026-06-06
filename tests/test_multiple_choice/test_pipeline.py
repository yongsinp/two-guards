import json
from unittest.mock import patch

from two_guards.core.loader import Document
from two_guards.core.config import Config, ModelConfig
from two_guards.multiple_choice.roles import run_generator, run_verifier
from two_guards.multiple_choice.pipeline import run


def test_run_generator_returns_parsed_output():
    gen_dict = {
        "fabricated_option": "The statute of limitations is 5 years.",
        "reasoning": "I changed the limitation period while keeping legal phrasing.",
    }
    with patch("two_guards.multiple_choice.roles.complete_json") as mock:
        gen_dict["reasoning_tokens"] = "I changed 3 years to 5 years."
        mock.return_value = gen_dict
        result = run_generator(
            document_text="The statute of limitations is 3 years.",
            hallucination_type="numerical_error",
            all_hallucination_types=["Temporal"],
            model="anthropic/claude-sonnet-4-5",
            reasoning_budget=8000,
        )

    assert result["fabricated_option"] == "The statute of limitations is 5 years."
    assert result["reasoning"] == "I changed the limitation period while keeping legal phrasing."
    assert result["reasoning_tokens"] == "I changed 3 years to 5 years."


def test_run_verifier_returns_parsed_output():
    verifier_dict = {
        "choice_indices": [1, 2],
        "reasoning": "Options 1 and 2 appear correct.",
    }
    with patch("two_guards.multiple_choice.roles.complete_json") as mock:
        verifier_dict["reasoning_tokens"] = "Checked each option against source"
        mock.return_value = verifier_dict
        result = run_verifier(
            document_text="The statute of limitations is 3 years.",
            options=["5 years", "3 years", "3-year limitation period"],
            model="anthropic/claude-sonnet-4-5",
            reasoning_budget=8000,
        )

    assert result["choice_indices"] == [1, 2]
    assert result["reasoning_tokens"] == "Checked each option against source"


def test_pipeline_passed_when_verifier_picks_any_option(tmp_path):
    """Verifier selecting any option means it was fooled (all options are lies)."""
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        reasoning_budget=8000,
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {
            "fabricated_option": "$15,000",
            "reasoning": "Increased monetary figure while preserving context.",
            "reasoning_tokens": "Changed amount.",
        },
        {
            "fabricated_option": "$20,000",
            "reasoning": "Inflated penalty for plausible confusion.",
            "reasoning_tokens": "Doubled amount.",
        },
    ]
    verifier_output = {
        "choice_indices": [1],
        "reasoning_tokens": "Index 1 looks plausible",
    }

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical", "magnitude"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        # Verifier selected an option → was fooled → passed
        assert call_kwargs["passed"] is True


def test_pipeline_passed_when_verifier_picks_multiple_options(tmp_path):
    """Verifier selecting multiple options still counts as fooled."""
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        reasoning_budget=8000,
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {
            "fabricated_option": "$15,000",
            "reasoning": "Adjusted amount to nearby plausible value.",
            "reasoning_tokens": "Changed.",
        },
        {
            "fabricated_option": "$20,000",
            "reasoning": "Magnified penalty without changing legal tone.",
            "reasoning_tokens": "Doubled.",
        },
    ]
    verifier_output = {
        "choice_indices": [0, 1],
        "reasoning_tokens": "Both look plausible",
    }

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical", "magnitude"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["passed"] is True


def test_pipeline_failed_when_verifier_picks_nothing(tmp_path):
    """Verifier selecting no options means it was not fooled → failed."""
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        reasoning_budget=8000,
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {
            "fabricated_option": "$15,000",
            "reasoning": "Modified numeric amount only.",
            "reasoning_tokens": "Changed.",
        },
    ]
    verifier_output = {
        "choice_indices": [],
        "reasoning_tokens": "None of these match the document",
    }

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        # Verifier picked nothing → not fooled → failed
        assert call_kwargs["passed"] is False
