"""Plan A pipeline: Adversarial Conversation (Liar → Verifier → Judge)."""

import random
from datetime import datetime, timezone

from two_guards.core.config import Config
from two_guards.core.loader import Document
from two_guards.core.writer import write_record
from two_guards.adversarial.roles import run_liar, run_verifier, run_judge


def run(config: Config, documents: list[Document]) -> None:
    """Run the adversarial conversation pipeline over a list of documents.

    For each document:
    - The Liar responds, randomly choosing to lie or tell the truth.
    - The Verifier attempts to detect any inconsistency.
    - If the Liar lied, a Judge determines whether the Verifier correctly
      identified the specific false claim.

    Records where the Liar lied and the Verifier failed to catch it are
    written to ``passed/``; all other records go to ``failed/``.

    Args:
        config: Project configuration (model names, paths, thinking budget).
        documents: Source documents to process.
    """
    for doc in documents:
        truth_flag = random.choice([True, False])

        liar_result = run_liar(
            document_text=doc.text,
            truth_flag=truth_flag,
            model=config.models.liar,
            budget_tokens=config.thinking.budget_tokens,
        )

        verifier_result = run_verifier(
            document_text=doc.text,
            liar_response=liar_result["response"],
            model=config.models.verifier,
            budget_tokens=config.thinking.budget_tokens,
        )

        if not liar_result["truth_flag"]:
            if verifier_result["attempted_correction"] and liar_result["false_claim"]:
                judge_result = run_judge(
                    false_claim=liar_result["false_claim"],
                    targeted_claim=verifier_result["targeted_claim"] or "",
                    model=config.models.judge,
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
