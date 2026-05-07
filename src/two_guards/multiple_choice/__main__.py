"""Entry point: python -m two_guards.multiple_choice."""

from two_guards.core.config import load_config
from two_guards.core.loader import load_documents
from two_guards.multiple_choice.pipeline import run
from two_guards.multiple_choice.prompts import HALLUCINATION_TYPE_REGISTRY


def main():
    config = load_config()
    documents = load_documents(config.input_dir)
    hallucination_types = list(HALLUCINATION_TYPE_REGISTRY.keys())
    if not hallucination_types:
        hallucination_types = ["generic"]
    run(config=config, documents=documents, hallucination_types=hallucination_types)


if __name__ == "__main__":
    main()
