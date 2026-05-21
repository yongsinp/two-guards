"""Entry point: python -m two_guards.multiple_choice."""

from two_guards.core.config import load_config, load_hallucination_types
from two_guards.core.loader import load_documents
from two_guards.multiple_choice.pipeline import run


def main():
    config = load_config()
    documents = load_documents(config.input_dir)
    hallucination_types = list(load_hallucination_types().keys())
    run(config=config, documents=documents, hallucination_types=hallucination_types)


if __name__ == "__main__":
    main()
