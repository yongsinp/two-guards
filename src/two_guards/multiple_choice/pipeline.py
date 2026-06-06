"""Plan B pipeline: Multiple Choice (Generators → Assemble → Verifier)."""

import random
from datetime import datetime, timezone

from tqdm import tqdm

from two_guards.core.config import Config
from two_guards.core.loader import Document
from two_guards.core.writer import write_record
from two_guards.multiple_choice.roles import run_generator, run_verifier


def run(config: Config, documents: list[Document], hallucination_types: list[str]) -> None:
    """Run the multiple-choice pipeline over a list of documents.

    For each document, one generator per hallucination type produces a
    fabricated statement option. These are shuffled and presented to a
    Verifier, which selects all statements it believes are correct (may be
    none). Since all options are fabricated lies, the verifier is expected
    to select nothing; any selection indicates a successful deception.

    Records where the Verifier selects at least one option are written to
    ``passed/`` (verifier was fooled); records where it selects nothing go
    to ``failed/``.

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
                all_hallucination_types=hallucination_types,
                model=config.models.generator,
                reasoning_budget=config.reasoning_budget,
                max_attempts=config.max_attempts,
            )
            generated.append(result)

        options = []
        for i, gen in enumerate(generated):
            options.append({
                "text": gen["fabricated_option"],
                "is_true": False,
                "hallucination_type": hallucination_types[i],
                "reasoning": gen.get("reasoning"),
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

        choice_indices = verifier_result.get("choice_indices", [])
        # All options are lies; any selection means the verifier was fooled
        passed = len(choice_indices) > 0

        record = {
            "plan": "B",
            "document_id": doc.id,
            "options": options,
            "verifier": {
                "choice_indices": choice_indices,
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
