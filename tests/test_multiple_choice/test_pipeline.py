import json
from unittest.mock import patch

from two_guards.core.loader import Document
from two_guards.core.config import Config, ModelConfig
from two_guards.multiple_choice.roles import run_generator, run_verifier
from two_guards.multiple_choice.pipeline import run


def test_run_generator_returns_parsed_output():
    gen_dict = {
        "fabricated_option": "The statute of limitations is 5 years.",
    }
    with patch("two_guards.multiple_choice.roles.complete_json") as mock:
        mock.return_value = (gen_dict, "I changed 3 years to 5 years.")
        result = run_generator(
            document_text="The statute of limitations is 3 years.",
            hallucination_type="numerical_error",
            model="anthropic/claude-sonnet-4-5",
            budget_tokens=8000,
        )

    assert result["fabricated_option"] == "The statute of limitations is 5 years."
    assert result["reasoning"] == "I changed 3 years to 5 years."


def test_run_verifier_returns_parsed_output():
    verifier_dict = {
        "choice_index": 2,
        "reasoning": "Option 2 matches the document.",
    }
    with patch("two_guards.multiple_choice.roles.complete_json") as mock:
        mock.return_value = (verifier_dict, None)
        result = run_verifier(
            document_text="The statute of limitations is 3 years.",
            options=["5 years", "7 years", "3 years"],
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["choice_index"] == 2


def test_pipeline_passed_when_verifier_picks_fabricated(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        budget_tokens=8000,
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {"fabricated_option": "$15,000", "reasoning": "Changed amount."},
        {"fabricated_option": "$20,000", "reasoning": "Doubled amount."},
    ]
    true_option_output = {"true_option": "$10,000"}
    verifier_output = {"choice_index": 1, "reasoning": "Seemed right."}

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_true_option", return_value=true_option_output), \
         patch("two_guards.multiple_choice.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical", "magnitude"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        # With shuffle mocked (no-op), options are: [true@0, fab1@1, fab2@2]
        # Verifier picked index 1 which is fabricated → passed
        assert call_kwargs["passed"] is True


def test_pipeline_failed_when_verifier_picks_true(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        budget_tokens=8000,
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {"fabricated_option": "$15,000", "reasoning": "Changed."},
    ]
    true_option_output = {"true_option": "$10,000"}
    verifier_output = {"choice_index": 0, "reasoning": "This is correct."}

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_true_option", return_value=true_option_output), \
         patch("two_guards.multiple_choice.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        # With shuffle mocked, options are: [true@0, fab1@1]
        # Verifier picked index 0 which is true → failed
        assert call_kwargs["passed"] is False
