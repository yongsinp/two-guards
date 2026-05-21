"""Configuration dataclasses and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "default.yaml"
_HALLUCINATION_TYPES_PATH = Path(__file__).parents[3] / "config" / "hallucination_types.yaml"


@dataclass
class ModelConfig:
    """Per-role model names using litellm's canonical format (provider/model).

    Each role can point to a different model or provider without code changes.
    """

    liar: str = "anthropic/claude-sonnet-4-6"
    verifier: str = "anthropic/claude-sonnet-4-6"
    generator: str = "anthropic/claude-sonnet-4-6"
    summarizer: str = "anthropic/claude-sonnet-4-6"
    tamperer: str = "anthropic/claude-sonnet-4-6"
    locator: str = "anthropic/claude-sonnet-4-6"
    judge: str = "anthropic/claude-sonnet-4-6"


@dataclass
class Config:
    """Top-level project configuration.

    Attributes:
        input_dir: Directory containing source .txt documents.
        output_dir: Root directory for JSONL output files.
        models: Per-role model assignments.
        budget_tokens: Maximum tokens to allocate for the reasoning trace.
        max_attempts: Maximum number of JSON parse attempts before raising an error.
    """

    input_dir: str = "data/input"
    output_dir: str = "data"
    models: ModelConfig = field(default_factory=ModelConfig)
    budget_tokens: int = 8000
    max_attempts: int = 3


def load_hallucination_types(path: Path | None = None) -> dict[str, str]:
    """Load hallucination types from YAML, returning a name→description mapping."""
    path = path or _HALLUCINATION_TYPES_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {name: entry["description"] for name, entry in raw.items()}


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file. Defaults to
            ``config/default.yaml`` relative to the project root.

    Returns:
        Populated Config instance.
    """
    path = path or _DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = ModelConfig(**raw.get("models", {}))
    return Config(
        input_dir=raw.get("input_dir", "data/input"),
        output_dir=raw.get("output_dir", "data"),
        models=models,
        budget_tokens=raw.get("budget_tokens", 8000),
        max_attempts=raw.get("max_attempts", 3),
    )
