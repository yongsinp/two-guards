"""Plan A pipeline: Adversarial Conversation (Liar → Verifier → Judge)."""

import random
from datetime import datetime, timezone

from tqdm import tqdm

from two_guards.adversarial.roles import run_liar, run_verifier, run_judge
from two_guards.core.config import Config, load_hallucination_types
from two_guards.core.loader import Document
from two_guards.core.writer import write_record

# temporary thresholds for now
MAX_UNCAUGHT_LIES: int = 3
MAX_TOTAL_LIES: int = 50


def run(config: Config, documents: list[Document]) -> None:
    """Run the adversarial conversation pipeline over a list of documents.

    For each document, we run a liar/verifier dialogue loop. In one turn of the loop:

    - The Liar produces one claim about the document, randomly choosing to lie or tell the truth. If it lies, it randomly chooses a specific
      hallucination type.
    - The Verifier attempts to detect any inconsistency.
    - An uncaught lie is one where the Liar lied (truth_flag=False) and the Verifier did not attempt a correction.
    - The loop ends when MAX_UNCAUGHT_LIES uncaught lies have accumulated OR
      MAX_TOTAL_CLAIMS liar turns have occurred.

    After the loop, the Judge receives the full transcript. For each conversation turn, it
    determines whether the Verifier correctly identified the specific false claim.

    Records where the Liar lied and the Verifier failed to catch it are
    written to ``passed/``; all other records go to ``failed/``.

    Args:
        config: Project configuration (model names, paths, thinking budget).
        documents: Source documents to process.
    """

    hallucination_types = load_hallucination_types()
    hallucination_types_keys = list(hallucination_types.keys())

    for doc in tqdm(documents):
        conversation: list[dict] = []
        num_uncaught_lies, total_lies, total_claims = 0, 0, 0
        last_verifier_response: str | None = None
        lies_so_far: list[str] = []

        while num_uncaught_lies < MAX_UNCAUGHT_LIES and total_lies < MAX_TOTAL_LIES:
            truth_flag = random.choices([True, False], weights=[0.4, 0.6])[0]
            liar_result = run_liar(
                document_text=doc.text,
                truth_flag=truth_flag,
                hallucination_type=None if truth_flag else random.choice(hallucination_types_keys),
                hallucination_types=hallucination_types,
                prior_verifier_response=last_verifier_response,
                lies_so_far=lies_so_far,
                model=config.models.liar,
                reasoning_budget=config.reasoning_budget,
                max_attempts=config.max_attempts,
            )

            verifier_result = run_verifier(
                document_text=doc.text,
                liar_response=liar_result["response"],
                model=config.models.verifier,
                reasoning_budget=config.reasoning_budget,
                max_attempts=config.max_attempts,
            )

            total_claims += 1
            lies_so_far.append(liar_result["false_claim"])
            last_verifier_response = verifier_result["response"]
            was_lie_uncaught_this_turn = liar_result["false_claim"] and not verifier_result.get("attempted_correction")
            if was_lie_uncaught_this_turn:
                num_uncaught_lies += 1

            turn: dict = {
                "turn_index": total_lies - 1,
                "liar": {
                    "claim": liar_result.get("claim"),
                    "truth_flag": liar_result.get("truth_flag"),
                    "hallucination_type": liar_result.get("hallucination_type"),
                    "false_claim": liar_result.get("false_claim"),
                    "reasoning_tokens": liar_result.get("reasoning_tokens"),
                },
                "verifier": {
                    "attempted_correction": verifier_result.get("attempted_correction"),
                    "targeted_claim": verifier_result.get("targeted_claim"),
                    "response": verifier_result.get("response"),
                    "reasoning_tokens": verifier_result.get("reasoning_tokens"),
                },
                "was_lie_uncaught_this_turn": was_lie_uncaught_this_turn,
            }

            judge_result = None
            if not liar_result["truth_flag"]:
                total_lies += 1
                if verifier_result["attempted_correction"] and liar_result["false_claim"]:
                    judge_result = run_judge(
                        document_text=doc.text,
                        turn_to_assess=turn,
                        model=config.models.judge,
                        reasoning_budget=config.reasoning_budget,
                        max_attempts=config.max_attempts,
                    )
                    judge_result["skipped"] = False

                if judge_result is None:
                    judge_result = {
                        "skipped": True,
                        "verifier_correctly_identified": False,
                        "did_liar_hallucinate": False,  # this is actually unknown
                        "undeclared_hallucinations": [],
                        "reasoning": (
                            "Verifier did not attempt a correction."
                            if not liar_result["truth_flag"]
                            else "Liar told truth; no false claim to identify."
                        ),
                        "reasoning_tokens": None,
                    }

            turn["judge"] = judge_result
            conversation.append(turn)

        lies_caught, undeclared_hallucination_turns = 0, 0
        # passed turns go into the dataset (uncaught lies), failed turns do not (truths & caught lies)
        passed_turns, failed_turns = [], []

        for turn in conversation:
            if turn["judge"]["verifier_correctly_identified"]:
                lies_caught += 1
            if turn["judge"]["did_liar_hallucinate"]:
                undeclared_hallucination_turns += 1

            if not turn["liar"]["truth_flag"] and (
                    turn["was_lie_uncaught_this_turn"] or not turn["judge"]["verifier_correctly_identified"]):
                passed_turns.append(turn)
            else:
                failed_turns.append(turn)

        judge_summary = {
            "lies_caught": lies_caught,
            "lies_missed": total_lies - lies_caught,
            "undeclared_hallucination_turns": undeclared_hallucination_turns,
        }

        base_record = {
            "plan": "A",
            "document_id": doc.id,
            "source_text": doc.text,
            "termination_reason": "max_uncaught_lies" if num_uncaught_lies >= MAX_UNCAUGHT_LIES else "max_total_lies",
            "total_claims": total_claims,
            "total_lies": total_lies,
            "num_uncaught_lies": num_uncaught_lies,
            "judge_summary": judge_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_record(
            output_dir=config.output_dir,
            plan="plan_a",
            passed=True,
            record={**base_record, "turns": passed_turns},
        )

        write_record(
            output_dir=config.output_dir,
            plan="plan_a",
            passed=False,
            record={**base_record, "turns": failed_turns},
        )
