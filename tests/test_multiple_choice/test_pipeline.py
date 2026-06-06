import json
from unittest.mock import patch

import pytest

from two_guards.core.loader import Document
from two_guards.core.config import Config, ModelConfig
from two_guards.core.llm import RateLimitError
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

        # 1 original record + 2 option-level records
        assert mock_write.call_count == 3
        calls = [c.kwargs for c in mock_write.call_args_list]
        original_call = next(c for c in calls if c.get("split") == "original")
        assert original_call["record"]["source_document"] == "The penalty is $10,000."
        option_calls = [c for c in calls if c.get("split") is None]
        assert len(option_calls) == 2
        assert sorted(c["passed"] for c in option_calls) == [False, True]


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

        assert mock_write.call_count == 3
        option_calls = [c.kwargs for c in mock_write.call_args_list if c.kwargs.get("split") is None]
        assert len(option_calls) == 2
        assert all(c["passed"] is True for c in option_calls)


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

        assert mock_write.call_count == 2
        calls = [c.kwargs for c in mock_write.call_args_list]
        original_call = next(c for c in calls if c.get("split") == "original")
        assert original_call["record"]["source_document"] == "The penalty is $10,000."
        option_call = next(c for c in calls if c.get("split") is None)
        assert option_call["passed"] is False


def test_pipeline_resumes_by_skipping_already_processed_documents(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        reasoning_budget=8000,
    )

    output_original = tmp_path / "output" / "plan_b" / "original"
    output_original.mkdir(parents=True)
    existing = {"plan": "B", "document_id": "doc_done"}
    (output_original / "plan_b_existing.jsonl").write_text(json.dumps(existing) + "\n", encoding="utf-8")

    docs = [
        Document(id="doc_done", text="Already done", source_path="a.txt"),
        Document(id="doc_new", text="Need processing", source_path="b.txt"),
    ]

    gen_outputs = [
        {
            "fabricated_option": "Incorrect statement",
            "reasoning": "Reasoning",
            "reasoning_tokens": "tokens",
        }
    ]
    verifier_output = {
        "choice_indices": [],
        "reasoning_tokens": "none selected",
    }

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=gen_outputs) as mock_gen, \
         patch("two_guards.multiple_choice.pipeline.run_verifier", return_value=verifier_output), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        run(config=config, documents=docs, hallucination_types=["numerical"])

        # Only doc_new should be processed
        mock_gen.assert_called_once()
        assert mock_gen.call_args[1]["document_text"] == "Need processing"
        assert mock_write.call_count == 2
        for call in mock_write.call_args_list:
            assert call.kwargs["record"]["document_id"] == "doc_new"


def test_pipeline_stops_on_rate_limit_to_allow_resume(tmp_path):
    config = Config(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        models=ModelConfig(),
        reasoning_budget=8000,
    )
    docs = [
        Document(id="doc_1", text="First", source_path="1.txt"),
        Document(id="doc_2", text="Second", source_path="2.txt"),
    ]

    with patch("two_guards.multiple_choice.pipeline.run_generator", side_effect=RateLimitError("limited")), \
         patch("two_guards.multiple_choice.pipeline.write_record") as mock_write, \
         patch("random.shuffle"):

        with pytest.raises(RateLimitError):
            run(config=config, documents=docs, hallucination_types=["numerical"])

        # No partial write for the interrupted document
        mock_write.assert_not_called()
