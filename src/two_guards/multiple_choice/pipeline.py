"""Plan B pipeline: Multiple Choice (Generators → Assemble → Judge)."""

import json
import random
from datetime import datetime, timezone

from two_guards.core.config import Config
from two_guards.core.llm import complete
from two_guards.core.loader import Document
from two_guards.core.writer import write_record
from two_guards.multiple_choice.roles import run_generator, run_judge
from two_guards.multiple_choice.prompts import TRUE_OPTION_SYSTEM, TRUE_OPTION_USER


def run_true_option(document_text: str, question: str, model: str) -> dict:
    """Generate the factually correct answer for a given question and document.

    Args:
        document_text: The source legal document.
        question: The question to answer.
        model: litellm model string.

    Returns:
        Dict with keys: true_option, question.
    """
    messages = [
        {"role": "system", "content": TRUE_OPTION_SYSTEM},
        {"role": "user", "content": TRUE_OPTION_USER.format(
            document_text=document_text,
            question=question,
        )},
    ]
    response = complete(model=model, messages=messages, thinking=False)
    return json.loads(response.content)


def run(config: Config, documents: list[Document], hallucination_types: list[str]) -> None:
    """Run the multiple-choice pipeline over a list of documents.

    For each document, one generator per hallucination type produces a
    fabricated answer option. These are assembled with the single true answer
    and shuffled. A Judge then selects what it believes is the correct answer.

    Records where the Judge picks a fabricated option are written to
    ``passed/``; records where it picks the true answer go to ``failed/``.

    Args:
        config: Project configuration (model names, paths, thinking budget).
        documents: Source documents to process.
        hallucination_types: List of hallucination type identifiers. One
            generator runs per type, producing one fabricated option each.
    """
    for doc in documents:
        generated = []
        question = None
        for h_type in hallucination_types:
            result = run_generator(
                document_text=doc.text,
                hallucination_type=h_type,
                model=config.models.generator,
                budget_tokens=config.thinking.budget_tokens,
            )
            generated.append(result)
            if question is None:
                question = result["question"]

        if not question:
            continue

        true_result = run_true_option(
            document_text=doc.text,
            question=question,
            model=config.models.judge,
        )

        options = []
        options.append({
            "text": true_result["true_option"],
            "is_true": True,
            "hallucination_type": None,
            "reasoning": None,
        })
        for i, gen in enumerate(generated):
            options.append({
                "text": gen["fabricated_option"],
                "is_true": False,
                "hallucination_type": hallucination_types[i],
                "reasoning": gen["reasoning"],
            })

        random.shuffle(options)

        option_texts = [opt["text"] for opt in options]
        judge_result = run_judge(
            document_text=doc.text,
            question=question,
            options=option_texts,
            model=config.models.judge,
        )

        choice_idx = judge_result["choice_index"]
        judge_correct = options[choice_idx]["is_true"] if 0 <= choice_idx < len(options) else False
        passed = not judge_correct

        record = {
            "plan": "B",
            "document_id": doc.id,
            "question": question,
            "options": options,
            "judge": {
                "choice_index": choice_idx,
                "correct": judge_correct,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_record(
            output_dir=config.output_dir,
            plan="plan_b",
            passed=passed,
            record=record,
        )
