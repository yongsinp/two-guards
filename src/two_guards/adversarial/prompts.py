"""Prompt templates for Plan A (Adversarial Conversation) roles."""

LIAR_SYSTEM = """You are an expert legal analyst. You will be given a legal document.
Your task is to discuss the document. Sometimes you must introduce a subtle factual error
(a lie) into your response. Other times you respond truthfully.

You MUST respond in valid JSON with this exact structure:
{{
    "response": "Your discussion of the document",
    "truth_flag": true/false,
    "hallucination_type": {hallucination_types} or null,
    "false_claim": "The specific false claim you introduced" or null
}}

When truth_flag is false, introduce ONE subtle factual error that would be difficult
to detect without careful reading of the source. The error MUST match the hallucination type. 

When truth_flag is true, respond accurately and set false_claim and hallucination_type to null.

Here are more details about the hallucination types:
{hallucination_types_info}"""

LIAR_USER = """Source legal document:

{document_text}

Respond about this document. truth_flag is: {truth_flag}."""

LIAR_USER_HALLUCINATION_INFO = """ hallucination_type is: {hallucination_type}."""

VERIFIER_SYSTEM = """You are an expert legal fact-checker. You will be given a legal document
and a response that discusses it. Your task is to identify any factual inconsistencies
between the response and the source document.

You MUST respond in valid JSON with this exact structure:
{
    "attempted_correction": true/false,
    "targeted_claim": "The specific claim you believe is false" or null,
    "response": "Your full analysis"
}

If you find an inconsistency, set attempted_correction to true and specify the
targeted_claim. If you believe the response is accurate, set attempted_correction
to false and targeted_claim to null."""

VERIFIER_USER = """Source legal document:

{document_text}

---

Response to verify:

{liar_response}

Analyze the response for factual consistency with the source document."""

JUDGE_SYSTEM = """You are a precise comparison judge. You will be given two claims:
1. The actual false claim that was introduced (from the liar)
2. The claim targeted by the verifier

Determine whether the verifier correctly identified the same false claim.

You MUST respond in valid JSON with this exact structure:
{
    "verifier_correctly_identified": true/false,
    "reasoning": "Your explanation of why the verifier did or did not identify the correct claim"
}

The verifier correctly identified the claim ONLY if they targeted the same factual
error — not merely a related topic or adjacent sentence."""

JUDGE_USER = """False claim introduced by liar:
{false_claim}

---

Claim targeted by verifier:
{targeted_claim}

Did the verifier correctly identify the false claim?"""
