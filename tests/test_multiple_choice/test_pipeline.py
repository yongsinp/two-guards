import json
from unittest.mock import patch

from two_guards.core.llm import LLMResponse
from two_guards.core.loader import Document
from two_guards.core.config import Config, ModelConfig, ThinkingConfig
from two_guards.multiple_choice.roles import run_generator, run_judge
from two_guards.multiple_choice.pipeline import run


def test_run_generator_returns_parsed_output():
    gen_json = json.dumps({
        "fabricated_option": "The statute of limitations is 5 years.",
        "question": "What is the statute of limitations?",
    })
    with patch("two_guards.multiple_choice.roles.complete") as mock:
        mock.return_value = LLMResponse(
            content=gen_json,
            reasoning="I changed 3 years to 5 years.",
        )
        result = run_generator(
            document_text="The statute of limitations is 3 years.",
            hallucination_type="numerical_error",
            model="anthropic/claude-sonnet-4-5",
            budget_tokens=8000,
        )

    assert result["fabricated_option"] == "The statute of limitations is 5 years."
    assert result["question"] == "What is the statute of limitations?"
    assert result["reasoning"] == "I changed 3 years to 5 years."


def test_run_judge_returns_parsed_output():
    judge_json = json.dumps({
        "choice_index": 2,
        "reasoning": "Option 2 matches the document.",
    })
    with patch("two_guards.multiple_choice.roles.complete") as mock:
        mock.return_value = LLMResponse(content=judge_json, reasoning=None)
        result = run_judge(
            document_text="The statute of limitations is 3 years.",
            question="What is the statute of limitations?",
            options=["5 years", "7 years", "3 years"],
            model="anthropic/claude-sonnet-4-5",
        )

    assert result["choice_index"] == 2


def test_pipeline_passed_when_judge_picks_fabricated(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        thinking=ThinkingConfig(budget_tokens=8000),
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {"fabricated_option": "$15,000", "question": "What is the penalty?", "reasoning": "Changed amount."},
        {"fabricated_option": "$20,000", "question": "What is the penalty?", "reasoning": "Doubled amount."},
    ]
    true_option_output = {"true_option": "$10,000", "question": "What is the penalty?"}
    judge_output = {"choice_index": 1, "reasoning": "Seemed right."}

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_true_option", return_value=true_option_output), \
         patch("two_guards.multiple_choice.pipeline.run_judge", return_value=judge_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical", "magnitude"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        # With shuffle mocked (no-op), options are: [true@0, fab1@1, fab2@2]
        # Judge picked index 1 which is fabricated → passed
        assert call_kwargs["passed"] is True


def test_pipeline_failed_when_judge_picks_true(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        thinking=ThinkingConfig(budget_tokens=8000),
    )
    doc = Document(id="doc_001", text="The penalty is $10,000.", source_path="test.txt")

    gen_outputs = [
        {"fabricated_option": "$15,000", "question": "What is the penalty?", "reasoning": "Changed."},
    ]
    true_option_output = {"true_option": "$10,000", "question": "What is the penalty?"}
    judge_output = {"choice_index": 0, "reasoning": "This is correct."}

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs), \
         patch("two_guards.multiple_choice.pipeline.run_true_option", return_value=true_option_output), \
         patch("two_guards.multiple_choice.pipeline.run_judge", return_value=judge_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=[doc], hallucination_types=["numerical"])

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args[1]
        # With shuffle mocked, options are: [true@0, fab1@1]
        # Judge picked index 0 which is true → failed
        assert call_kwargs["passed"] is False
