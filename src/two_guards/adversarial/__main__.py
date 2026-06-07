"""Entry point: python -m two_guards.adversarial."""

from two_guards.adversarial.pipeline import run
from two_guards.core.config import load_config
from two_guards.core.loader import load_documents


def main():
    config = load_config()
    documents = load_documents(config.input_dir)
    run(config=config, documents=documents)


if __name__ == "__main__":
    main()
