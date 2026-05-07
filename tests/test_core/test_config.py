import tempfile
from pathlib import Path

from two_guards.core.config import load_config


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
thinking:
  budget_tokens: 8000
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)

    assert config.input_dir == "data/input"
    assert config.output_dir == "data"
    assert config.models.liar == "anthropic/claude-sonnet-4-6"
    assert config.models.judge == "anthropic/claude-sonnet-4-6"
    assert config.thinking.budget_tokens == 8000


def test_config_default_path():
    config = load_config()
    assert config.input_dir is not None
    assert config.models.liar is not None
