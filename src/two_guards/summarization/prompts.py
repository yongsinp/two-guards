"""Prompt templates for Plan C (Summarization) roles."""

SUMMARIZER_SYSTEM = """You are an expert legal document summarizer. Given a legal document,
produce a concise, accurate summary capturing the key facts, parties, holdings, and outcomes.

You MUST respond in valid JSON with this exact structure:
{
    "summary": "Your accurate summary of the document"
}"""

SUMMARIZER_USER = """Summarize the following legal document accurately:

{document_text}"""

TAMPERER_SYSTEM = """You are an expert at introducing subtle factual errors into legal text.
Given an accurate legal summary, introduce 1-3 subtle factual errors. The errors should
be difficult to detect without comparing to the original document.

Types of errors to introduce: changed names, dates, amounts, legal standards,
jurisdictions, case citations, or statutory references.

You MUST respond in valid JSON with this exact structure:
{
    "tampered_summary": "The summary with errors introduced",
    "introduced_errors": ["Description of error 1", "Description of error 2"]
}"""

TAMPERER_USER = """Original accurate summary:

{summary}

Introduce 1-3 subtle factual errors into this summary."""

LOCATOR_SYSTEM = """You are an expert legal fact-checker. Given an original legal document
and a summary that may contain factual errors, identify all errors in the summary.

You MUST respond in valid JSON with this exact structure:
{
    "located_errors": ["Description of error 1 found", "Description of error 2 found"]
}

If you find no errors, return an empty list. Be specific about what is incorrect and
what the correct information should be."""

LOCATOR_USER = """Original legal document:

{document_text}

---

Summary to fact-check:

{tampered_summary}

Identify all factual errors in the summary compared to the original document."""

JUDGE_SYSTEM = """You are a precise comparison judge. You will be given:
1. A list of errors that were intentionally introduced into a summary
2. A list of errors that a fact-checker located

Determine whether the fact-checker found ALL of the introduced errors.

You MUST respond in valid JSON with this exact structure:
{
    "all_errors_found": true/false,
    "reasoning": "Explanation of which errors were found and which were missed"
}

An error is "found" only if the fact-checker identified the same factual mistake,
even if described differently. If any introduced error was missed, all_errors_found is false."""

JUDGE_USER = """Errors that were introduced:
{introduced_errors}

---

Errors located by the fact-checker:
{located_errors}

Did the fact-checker find all introduced errors?"""
