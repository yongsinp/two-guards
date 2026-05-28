"""Plan B pipeline: Multiple Choice (Generators → Assemble → Verifier)."""

import random
from datetime import datetime, timezone

from tqdm import tqdm

from two_guards.core.config import Config
from two_guards.core.llm import complete_json
from two_guards.core.loader import Document
from two_guards.core.writer import write_record
from two_guards.multiple_choice.roles import run_generator, run_verifier
from two_guards.multiple_choice.prompts import TRUE_OPTION_SYSTEM, TRUE_OPTION_USER


def run_true_option(document_text: str, model: str, max_attempts: int = 3) -> dict:
    """Generate one factually correct statement for a given document.

    Args:
        document_text: The source legal document.
        model: litellm model string.
        max_attempts: Maximum number of retry attempts for JSON parsing.

    Returns:
        Dict with key: true_option.
    """
    messages = [
        {"role": "system", "content": TRUE_OPTION_SYSTEM},
        {"role": "user", "content": TRUE_OPTION_USER.format(document_text=document_text)},
    ]
    return complete_json(model=model, messages=messages, max_attempts=max_attempts)


def run(config: Config, documents: list[Document], hallucination_types: list[str]) -> None:
    """Run the multiple-choice pipeline over a list of documents.

    For each document, one generator per hallucination type produces a
    fabricated statement option. These are assembled with one true statement
    and shuffled. A Verifier then selects what it believes is the correct statement.

    Records where the Verifier picks a fabricated option are written to
    ``passed/``; records where it picks the true statement go to ``failed/``.

    Args:
        config: Project configuration (model names, paths, thinking budget).
        documents: Source documents to process.
        hallucination_types: List of hallucination type identifiers. One
            generator runs per type, producing one fabricated option each.
    """
    for doc in tqdm(documents):
        generated = []
        for h_type in hallucination_types:
            result = run_generator(
                document_text=doc.text,
                hallucination_type=h_type,
                model=config.models.generator,
                reasoning_budget=config.reasoning_budget,
                max_attempts=config.max_attempts,
            )
            generated.append(result)

        true_result = run_true_option(
            document_text=doc.text,
            model=config.models.verifier,
            max_attempts=config.max_attempts,
        )

        options = []
        options.append({
            "text": true_result["true_option"],
            "is_true": True,
            "hallucination_type": None,
            "reasoning_tokens": None,
        })
        for i, gen in enumerate(generated):
            options.append({
                "text": gen["fabricated_option"],
                "is_true": False,
                "hallucination_type": hallucination_types[i],
                "reasoning_tokens": gen.get("reasoning_tokens"),
            })

        random.shuffle(options)

        option_texts = [opt["text"] for opt in options]
        verifier_result = run_verifier(
            document_text=doc.text,
            options=option_texts,
            model=config.models.verifier,
            reasoning_budget=config.reasoning_budget,
            max_attempts=config.max_attempts,
        )

        choice_idx = verifier_result["choice_index"]
        verifier_correct = options[choice_idx]["is_true"] if 0 <= choice_idx < len(options) else False
        passed = not verifier_correct

        record = {
            "plan": "B",
            "document_id": doc.id,
            "options": options,
            "verifier": {
                "choice_index": choice_idx,
                "correct": verifier_correct,
                "reasoning_tokens": verifier_result.get("reasoning_tokens"),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_record(
            output_dir=config.output_dir,
            plan="plan_b",
            passed=passed,
            record=record,
        )
