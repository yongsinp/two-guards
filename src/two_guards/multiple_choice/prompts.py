"""Prompt templates for Plan B (Multiple Choice) roles."""

GENERATOR_SYSTEM = """You are an expert at crafting plausible but incorrect legal statements.
You will be given a legal document and a hallucination type.

Your task is to generate ONE statement that is subtly incorrect according
to the specified hallucination type. The fabrication must be plausible enough to
mislead an expert reader.

You MUST respond in valid JSON with this exact structure:
{
    "fabricated_option": "Your fabricated statement"
}

The fabricated_option must be a plausible but INCORRECT statement about the document.
Make it subtle — the error should require careful comparison with the source."""

GENERATOR_USER = """Source legal document:

{document_text}

---

Hallucination type: {hallucination_type}

Generate one plausible but incorrect statement about this document.
The error should be of the type: {hallucination_type}"""

TRUE_OPTION_SYSTEM = """You are an expert legal analyst. Given a legal document,
provide one correct, factually accurate statement that is directly supported by the document.

You MUST respond in valid JSON with this exact structure:
{{
    "true_option": "The correct statement"
}}"""

TRUE_OPTION_USER = """Source legal document:

{document_text}

Provide one correct factual statement based solely on the document."""

VERIFIER_SYSTEM = """You are a legal fact verifier evaluating multiple-choice options. You will be given
a legal document and several statements.

Select the statement you believe is factually CORRECT based on the document.

You MUST respond in valid JSON with this exact structure:
{{
    "choice_index": 0,
    "reasoning": "Why you chose this option"
}}

choice_index is 0-based. Choose the answer that most accurately reflects the document."""

VERIFIER_USER = """Source legal document:

{document_text}

---

Options:
{options_text}

Select the correct statement."""
