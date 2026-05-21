from pathlib import Path

from two_guards.core.config import load_config, load_hallucination_types


def test_load_config_from_yaml(tmp_path: Path):
    yaml_content = """
input_dir: "data/input"
output_dir: "data"
models:
  liar: "anthropic/claude-sonnet-4-6"
  verifier: "anthropic/claude-sonnet-4-6"
  generator: "anthropic/claude-sonnet-4-6"
  summarizer: "anthropic/claude-sonnet-4-6"
  tamperer: "anthropic/claude-sonnet-4-6"
  locator: "anthropic/claude-sonnet-4-6"
  judge: "anthropic/claude-sonnet-4-6"
budget_tokens: 8000
max_attempts: 3
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)

    assert config.input_dir == "data/input"
    assert config.output_dir == "data"
    assert config.models.liar == "anthropic/claude-sonnet-4-6"
    assert config.models.judge == "anthropic/claude-sonnet-4-6"
    assert config.budget_tokens == 8000
    assert config.max_attempts == 3


def test_config_default_path():
    config = load_config()
    assert config.input_dir is not None
    assert config.models.liar is not None
    assert config.budget_tokens == 8000
    assert config.max_attempts == 3


def test_load_hallucination_types_from_yaml(tmp_path: Path):
    yaml_content = """
Temporal:
    description: "Time-sensitive errors."
    example: "An example."

Relation:
    description: "Incorrect entity relationships."
    example: "Another example."
"""
    types_file = tmp_path / "hallucination_types.yaml"
    types_file.write_text(yaml_content)

    result = load_hallucination_types(types_file)

    assert result == {
        "Temporal": "Time-sensitive errors.",
        "Relation": "Incorrect entity relationships.",
    }


def test_load_hallucination_types_default_path():
    result = load_hallucination_types()
    assert len(result) > 0
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in result.items())
