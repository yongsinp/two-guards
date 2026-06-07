"""Plan B pipeline: Multiple Choice (Generators → Assemble → Verifier)."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from two_guards.core.config import Config
from two_guards.core.loader import Document
from two_guards.core.llm import RateLimitError
from two_guards.core.writer import write_record
from two_guards.multiple_choice.roles import run_liar, run_verifier


def _load_processed_document_ids(output_dir: str) -> set[str]:
    """Collect already-written Plan B document IDs from original outputs."""
    root = Path(output_dir) / "plan_b" / "original"
    processed_ids: set[str] = set()
    if not root.exists():
        return processed_ids

    for jsonl_file in root.glob("**/*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = record.get("document_id")
                if doc_id:
                    processed_ids.add(doc_id)

    return processed_ids


def run(config: Config, documents: list[Document], hallucination_types: list[str]) -> None:
    """Run the multiple-choice pipeline over a list of documents.

    For each document, one liar call per hallucination type produces a
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
            liar runs per type, producing one fabricated option each.
    """
    processed_document_ids = _load_processed_document_ids(config.output_dir)

    for doc in tqdm(documents):
        if doc.id in processed_document_ids:
            continue

        generated = []
        try:
            for h_type in hallucination_types:
                result = run_liar(
                    document_text=doc.text,
                    hallucination_type=h_type,
                    all_hallucination_types=hallucination_types,
                    model=config.models.liar,
                    reasoning_budget=config.reasoning_budget,
                    max_attempts=config.max_attempts,
                )
                generated.append(result)

            options = []
            for i, gen in enumerate(generated):
                options.append({
                    "text": gen["fabricated_option"],
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
            original_record = {
                "plan": "B",
                "document_id": doc.id,
                "source_document": doc.text,
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
                passed=None,
                split="original",
                record=original_record,
            )

            selected_indices = {
                idx for idx in choice_indices if isinstance(idx, int) and 0 <= idx < len(options)
            }
            for i, option in enumerate(options):
                option_record = {
                    "plan": "B",
                    "document_id": doc.id,
                    "source_document": doc.text,
                    "option_index": i,
                    "option": option,
                    "selected_by_verifier": i in selected_indices,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                write_record(
                    output_dir=config.output_dir,
                    plan="plan_b",
                    passed=i in selected_indices,
                    record=option_record,
                )

            processed_document_ids.add(doc.id)
        except RateLimitError:
            raise
