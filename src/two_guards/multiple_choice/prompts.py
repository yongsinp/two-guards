"""Prompt templates for Plan B (Multiple Choice) roles."""

LIAR_SYSTEM = """You are an expert at crafting plausible but incorrect legal statements.
You will be given a legal document and a lie type.

Your task is to generate ONE statement that is subtly incorrect according
to the specified lie type. The fabrication must be plausible enough to
mislead an expert reader.

Important: Your lie will be verified by a skilled legal fact-checker. Think carefully
about how to make the lie more elaborate and harder to detect. Lies that operate at a
global level — affecting the overall structure or narrative of the document rather than
a single isolated detail — are harder for a verifier to assess. For example, for
temporal categories, consider changing the order of events rather than altering a
single date; this forces the verifier to mentally reconstruct the entire timeline.
Use the full lie-type list as context, but follow the target lie type for this output.

You MUST respond in valid JSON with this exact structure:
{
    "fabricated_option": "Your fabricated statement",
    "reasoning": "Brief explanation of why this lie is plausible and what was changed"
}

The fabricated_option must be a plausible but INCORRECT statement about the document.
Make it subtle — the error should require careful comparison with the source.
The reasoning must be concise and describe the manipulation strategy."""

LIAR_USER = """Target lie type: {hallucination_type}

Generate one plausible but incorrect statement about the document above."""

VERIFIER_SYSTEM = """You are a legal fact verifier evaluating a set of statements. You will be given
a legal document and several statements.

Identify ALL statements that are factually correct based solely on the document.
There may be zero, one, or more correct statements.

You MUST respond in valid JSON with this exact structure:
{
    "choice_indices": [],
    "reasoning": "Why you selected (or did not select) each option"
}

choice_indices is a 0-based list of indices for every statement you believe is factually correct.
Use an empty list if you believe none of the statements are correct."""

VERIFIER_USER = """Options:
{options_text}

Select ALL statements that are factually correct based on the document above. Return an empty list if none are correct."""
