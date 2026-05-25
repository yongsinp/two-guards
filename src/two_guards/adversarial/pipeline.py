"""Plan A pipeline: Adversarial Conversation (Liar → Verifier → Judge)."""

import random
from datetime import datetime, timezone

from two_guards.adversarial.roles import run_liar, run_verifier, run_judge
from two_guards.core.config import Config, load_hallucination_types
from two_guards.core.loader import Document
from two_guards.core.writer import write_record


def run(config: Config, documents: list[Document]) -> None:
    """Run the adversarial conversation pipeline over a list of documents.

    For each document:
    - The Liar responds, randomly choosing to lie or tell the truth. If it lies, it randomly chooses a specific
      hallucination type.
    - The Verifier attempts to detect any inconsistency.
    - If the Liar lied, a Judge determines whether the Verifier correctly
      identified the specific false claim.

    Records where the Liar lied and the Verifier failed to catch it are
    written to ``passed/``; all other records go to ``failed/``.

    Args:
        config: Project configuration (model names, paths, thinking budget).
        documents: Source documents to process.
    """

    hallucination_types = load_hallucination_types()
    hallucination_types_keys = list(hallucination_types.keys())

    for doc in documents:
        truth_flag = random.choices([True, False], weights=[0.4, 0.6])
        hallucination_type = None if truth_flag else random.choice(hallucination_types_keys)
        liar_result = run_liar(
            document_text=doc.text,
            truth_flag=truth_flag,
            hallucination_type=hallucination_type,
            hallucination_type_info=None if truth_flag else hallucination_types[hallucination_type],
            model=config.models.liar,
            budget_tokens=config.budget_tokens,
            max_attempts=config.max_attempts,
        )

        verifier_result = run_verifier(
            document_text=doc.text,
            liar_response=liar_result["response"],
            model=config.models.verifier,
            budget_tokens=config.budget_tokens,
            max_attempts=config.max_attempts,
        )

        if not liar_result["truth_flag"]:
            if verifier_result["attempted_correction"] and liar_result["false_claim"]:
                judge_result = run_judge(
                    false_claim=liar_result["false_claim"],
                    targeted_claim=verifier_result["targeted_claim"] or "",
                    model=config.models.judge,
                    max_attempts=config.max_attempts,
                )
                passed = not judge_result["verifier_correctly_identified"]
            else:
                judge_result = {
                    "verifier_correctly_identified": False,
                    "reasoning": "Verifier did not attempt a correction.",
                }
                passed = True
        else:
            judge_result = {
                "verifier_correctly_identified": False,
                "reasoning": "Liar told truth; no false claim to identify.",
            }
            passed = False

        record = {
            "plan": "A",
            "document_id": doc.id,
            "source_text": doc.text,
            "liar": {
                "response": liar_result["response"],
                "reasoning": liar_result["reasoning"],
                "truth_flag": liar_result["truth_flag"],
                "hallucination_type": liar_result.get("hallucination_type"),
                "false_claim": liar_result.get("false_claim"),
            },
            "verifier": {
                "response": verifier_result["response"],
                "reasoning": verifier_result["reasoning"],
                "attempted_correction": verifier_result["attempted_correction"],
                "targeted_claim": verifier_result.get("targeted_claim"),
            },
            "judge": judge_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_record(
            output_dir=config.output_dir,
            plan="plan_a",
            passed=passed,
            record=record,
        )
