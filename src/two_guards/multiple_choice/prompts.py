"""Prompt templates for Plan B (Multiple Choice) roles and hallucination type registry."""

GENERATOR_SYSTEM = """You are an expert at crafting plausible but incorrect legal statements.
You will be given a legal document and a hallucination type.

Your task is to generate ONE multiple-choice option that is subtly incorrect according
to the specified hallucination type. The fabrication must be plausible enough to
mislead an expert reader.

You MUST respond in valid JSON with this exact structure:
{{
    "fabricated_option": "Your fabricated answer text",
    "question": "A question about the document that this option answers"
}}

The fabricated_option must be a plausible but INCORRECT answer to the question.
Make it subtle — the error should require careful comparison with the source."""

GENERATOR_USER = """Source legal document:

{document_text}

---

Hallucination type: {hallucination_type}

Generate a plausible but incorrect option for a multiple-choice question about this document.
The error should be of the type: {hallucination_type}"""

TRUE_OPTION_SYSTEM = """You are an expert legal analyst. Given a legal document and a question,
provide the correct, factually accurate answer.

You MUST respond in valid JSON with this exact structure:
{{
    "true_option": "The correct answer",
    "question": "The question (repeat it back)"
}}"""

TRUE_OPTION_USER = """Source legal document:

{document_text}

---

Question: {question}

Provide the correct answer to this question based solely on the document."""

JUDGE_SYSTEM = """You are taking a multiple-choice legal knowledge test. You will be given
a legal document, a question about it, and several answer options.

Select the option you believe is CORRECT based on the document.

You MUST respond in valid JSON with this exact structure:
{{
    "choice_index": 0,
    "reasoning": "Why you chose this option"
}}

choice_index is 0-based. Choose the answer that most accurately reflects the document."""

JUDGE_USER = """Source legal document:

{document_text}

---

Question: {question}

Options:
{options_text}

Select the correct answer."""

HALLUCINATION_TYPE_REGISTRY: dict[str, str] = {}
