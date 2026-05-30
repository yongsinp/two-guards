#!/usr/bin/env python3
"""
CourtListener SCOTUS Opinion Downloader (2025–2026)
====================================================
Downloads Supreme Court opinion PDFs filed between 2025-01-01 and 2026-12-31,
and converts each PDF to a clean .txt file with:
  - Page headers / running titles stripped
  - Standalone page numbers removed
  - Footnotes collected and appended at the end

Usage:
    python download_files.py --token YOUR_API_TOKEN [options]

Options:
    --token     API token from courtlistener.com/profile/  (required)
    --out-dir   Directory to save TXT files (default: ./scotus_opinions)
    --from-date Start date YYYY-MM-DD (default: 2025-01-01)
    --to-date   End date   YYYY-MM-DD (default: 2026-12-31)
    --limit     Max number of opinions to download (default: unlimited)
    --dry-run   List cases without downloading files
"""

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import requests
from pypdf import PdfReader

# spellchecker
from spellchecker import SpellChecker

BASE_URL    = "https://www.courtlistener.com"
SEARCH_URL  = f"{BASE_URL}/api/rest/v4/search/"
STORAGE_URL = "https://storage.courtlistener.com"

# Polite delay between PDF downloads (seconds) — respect rate limits
DOWNLOAD_DELAY = 0.5


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def build_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Token {token}",
        "User-Agent": "scotus-downloader/1.0 (research use)",
    })
    return s


def search_opinions(session: requests.Session, from_date: str, to_date: str):
    """
    Generator that yields every opinion result from the Search API,
    handling cursor-based pagination automatically.
    """
    params = {
        "type":         "o",
        "court":        "scotus",
        "filed_after":  from_date,
        "filed_before": to_date,
        "order_by":     "dateFiled asc",
        "page_size":    20,
    }

    url  = SEARCH_URL
    page = 0

    while url:
        page += 1
        print(f"  [page {page}] Fetching search results …", end="", flush=True)

        resp = session.get(url, params=params if page == 1 else None)

        if resp.status_code == 401:
            sys.exit("\n✗ 401 Unauthorized — check your API token.")
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(f"\n  Rate-limited. Waiting {retry_after}s …")
            time.sleep(retry_after)
            continue
        resp.raise_for_status()

        data    = resp.json()
        results = data.get("results", [])
        print(f" {len(results)} results (total found: {data.get('count', '?')})")

        yield from results

        url    = data.get("next")
        params = None


def get_pdf_url(opinion: dict) -> "str | None":
    """
    Extract the best available PDF URL from a search result opinion entry.
    Falls back through: local_path → download_url → None
    """
    for op in opinion.get("opinions", []):
        local = op.get("local_path")
        if local:
            return f"{STORAGE_URL}/{local}"
        download = op.get("download_url")
        if download and download.lower().endswith(".pdf"):
            return download
    return None

# ---------------------------------------------------------------------------
# SCOTUS-specific text cleaning
# ---------------------------------------------------------------------------

spell = SpellChecker()

def fix_mid_word_spaces (text: str) -> str:
    # fixes spaces inserted into words seemingly randomly from PDF conversion
    # ex: from Royal Canin case; "clai ms" -> "claims", "o riginal" -> "original"
    def merge_if_word(m):
        left = m.group(1)
        right = m.group(2)
        merged = left+right

        # leaves it alone if both words are valid
        # ex: "the" "rapist" doesn't become "therapist" every time
        if left in spell and right in spell:
            return m.group(0)

        # checks to see if two words merged can be one word
        if merged in spell:
            return merged
        return m.group(0)
    return re.sub(r"([a-z]+) ([a-z])", merge_if_word, text)

# ── Footnote detection ──────────────────────────────────────────────────────
# A footnote block starts with a horizontal rule (—— or similar dashes/em-dashes)
# followed on the next (non-blank) line by a digit, and runs until the next such
# divider or end-of-text.
_FOOTNOTE_SEP = re.compile(
    r"(?:^|\n)"                     # start of string or newline
    r"[ \t]*[—\-–]{2,}[ \t]*\n"    # line of dashes / em-dashes (the rule)
    r"((?:.|\n)*?)"                 # footnote content (non-greedy)
    r"(?=[ \t]*[—\-–]{2,}[ \t]*\n|$)",  # lookahead: next rule OR end
    re.MULTILINE,
)

# Matches a line that is ONLY a page number (possibly with surrounding spaces)
_LONE_PAGE_NUM = re.compile(r"(?m)^\s*\d+\s*$")

# SCOTUS running headers come in two flavours:
#
#   ODD pages (after page 1):
#     Cite as: 608 U. S. ___ (2026)
#     [optional blank]
#     <section label>        ← e.g. "Syllabus", "Opinion of the Court", "THOMAS, J., dissenting"
#
#   EVEN pages:
#     CASE NAME v.\n
#     RESPONDENT\n
#     <section label>
#
# We also need to strip the slip-opinion banner that appears at the very top:
#   (Slip Opinion)
#   OCTOBER TERM, 20XX

_SLIP_BANNER = re.compile(
    r"^\(Slip Opinion\)\s*\n"
    r"OCTOBER TERM,\s*\d{4}\s*\n",
    re.MULTILINE,
)

# "Cite as: NNN U. S. ___ (YYYY)" line (odd-page header line 1)
_CITE_AS = re.compile(
    r"(?m)^Cite as:\s*\d+\s*U\.\s*S\.\s*[_\d]+\s*\(\d{4}\)\s*$"
)

# Section-label lines that appear as the last line of a running header block.
# These are short ALL-CAPS or title-case labels we recognise explicitly.
_SECTION_LABELS = re.compile(
    r"(?m)^("
    r"Syllabus"
    r"|Opinion of the Court"
    r"|OPINION OF THE COURT"
    r"|Per Curiam"
    r"|PER CURIAM"
    r"|[A-Z][A-Z .,']+,\s*J\.,\s*(?:dissenting|concurring|concurring in (?:part|the judgment)|dissenting in part)"
    r"|[A-Z][A-Z .,']+,\s*C\.\s*J\.,\s*(?:dissenting|concurring|concurring in (?:part|the judgment)|dissenting in part)"
    r")\s*$"
)

# All-caps case-name lines typical of even-page headers, e.g.:
#   FIRST CHOICE WOMEN'S RESOURCE CENTERS, INC. v.
#   DAVENPORT
# These are ≤ 2 consecutive ALL-CAPS lines (with possible punctuation).
_ALLCAPS_HEADER = re.compile(
    r"(?m)^([A-Z][A-Z0-9 ,.'()&\-]+(?:\s+v\.)?)\s*$"
)


def _extract_footnotes(text: str) -> tuple[str, list[str]]:
    """
    Scan *text* for footnote blocks delimited by a line of em-dashes / hyphens.
    Returns (body_without_footnotes, [footnote_text, …]).
    """
    footnotes: list[str] = []
    positions_to_remove: list[tuple[int, int]] = []

    # Find every separator line and its position
    sep_pattern = re.compile(r"[ \t]*[—\-–]{2,}[ \t]*\n", re.MULTILINE)
    separators  = list(sep_pattern.finditer(text))

    for i, sep in enumerate(separators):
        # Content runs from end of this separator to start of the next (or EOF)
        content_start = sep.end()
        content_end   = separators[i + 1].start() if i + 1 < len(separators) else len(text)
        block         = text[content_start:content_end].strip()

        if not block:
            continue

        # Only treat as footnote if the block starts with a digit (footnote number)
        if re.match(r"^\d", block):
            footnotes.append(block)
            positions_to_remove.append((sep.start(), content_end))

    # Remove footnote blocks from the body (work backwards to preserve offsets)
    for start, end in reversed(positions_to_remove):
        text = text[:start] + text[end:]

    return text, footnotes


def _remove_page_headers(text: str) -> str:
    """
    Remove SCOTUS running-header artefacts that PyPDF2 extracts at the top of
    each page.  Headers are stripped heuristically line-by-line.
    """
    lines   = text.split("\n")
    out     = []
    i       = 0
    n       = len(lines)

    while i < n:
        line = lines[i]

        # ── Slip-opinion banner (2 lines) ───────────────────────────────────
        if re.match(r"^\(Slip Opinion\)\s*$", line) and i + 1 < n and re.match(
            r"^OCTOBER TERM,\s*\d{4}\s*$", lines[i + 1]
        ):
            i += 2
            continue

        # ── Lone page number ────────────────────────────────────────────────
        if re.match(r"^\s*\d+\s*$", line):
            # Peek at the next non-blank line; if it looks like a header, skip
            # the page number too.
            i += 1
            continue

        # ── "Cite as: …" (odd-page header, line 1) ─────────────────────────
        if re.match(r"^Cite as:\s*\d+\s*U\.\s*S\.\s*[_\d]+\s*\(\d{4}\)\s*$", line):
            # Skip this line plus any immediately following section label
            i += 1
            while i < n and re.match(r"^\s*$", lines[i]):
                i += 1
            if i < n and _SECTION_LABELS.match(lines[i]):
                i += 1
            continue

        # ── ALL-CAPS case-name header lines (even-page header) ──────────────
        # Heuristic: an all-caps line of ≥ 10 chars that is NOT the start of a
        # known section is likely an even-page case-name header.
        if (
            re.match(r"^[A-Z][A-Z0-9 ,.'()&\-]{9,}$", line)
            and not _SECTION_LABELS.match(line)
        ):
            # Could be 1 or 2 header lines followed by a section label
            j = i + 1
            while j < n and re.match(r"^[A-Z][A-Z0-9 ,.'()&\-]*$", lines[j]):
                j += 1
            # If the line(s) are followed by a section label, drop them all
            if j < n and _SECTION_LABELS.match(lines[j]):
                i = j + 1
                continue
            # If there's a trailing "v." it's definitely a case name line
            if line.rstrip().endswith(" v."):
                i = j
                continue

        # ── Section-label lines that appear alone (e.g. "Syllabus") ─────────
        # Keep these — they are part of the opinion body.

        out.append(line)
        i += 1

    return "\n".join(out)

def clean_scotus_text(raw: str) -> tuple[str, list[str]]:
    """
    Full cleaning pipeline for a raw SCOTUS PDF text dump.

    Returns:
        body       : cleaned opinion text (string)
        footnotes  : list of footnote strings, in document order
    """

    # 1. Normalise line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 1.5. Strip headers
    text = _remove_page_headers(text)


    # 2. Extract footnotes (delimited by —— lines) before other cleaning
    #    so we don't accidentally mangle them.
    text, footnotes = _extract_footnotes(text)
    #footnotes = []
    #text = "SEEHERE" + text

    # 3. Strip page headers / running titles
  #  text = _remove_page_headers(text)
    #
    # 4. Remove any residual lone page numbers
    text = _LONE_PAGE_NUM.sub("", text)

    # 4.5. Fix mid word spaces
    text = fix_mid_word_spaces(text)
    #
    # 5. Collapse excessive blank lines
    #text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    #text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    #
    # 6. Join soft line-breaks within paragraphs (single newline → space),
    #    but preserve intentional paragraph breaks (double newline).
    #text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", " ", text)

    #
    # 7. Tidy multiple spaces
    text = re.sub(r" {2,}", " ", text)
    #

    # 7.5 clean footnotes
    cleaned_footnotes = []
    for fn in footnotes:
        fn = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", fn)
        fn = fix_mid_word_spaces(fn)
        fn = re.sub(r"\n+", " ", fn)
        fn = re.sub(r" {2,}", " ", fn)
        cleaned_footnotes.append(fn.strip())

    # 8. Strip each paragraph
    paragraphs = [p.strip() for p in text.split("\n\n")]
    text = "\n\n".join(p for p in paragraphs if p)

    return text, cleaned_footnotes


def assemble_output(body: str, footnotes: list[str]) -> str:
    """Combine body + footnotes section into the final .txt content."""
    parts = [body]
    if footnotes:
        parts.append("\n" + "=" * 72)
        parts.append("FOOTNOTES")
        parts.append("=" * 72 + "\n")
        for fn in footnotes:
            # Ensure each footnote is separated by a blank line
            parts.append(fn.strip())
            parts.append("")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def safe_basename(case_name: str, opinion_id: int) -> str:
    """Build a safe base name (no extension) from the case name and opinion ID."""
    clean = "".join(c if c.isalnum() or c in " -_" else "_" for c in case_name)
    clean = clean.strip().replace("  ", " ")[:80]
    return f"{opinion_id}_{clean}"

def fetch_and_convert(session: requests.Session, url: str, pdf_dest: Path, txt_dest: Path) -> bool:
    """
    Fetch a PDF, save it to disk, extract + clean its text (SCOTUS-aware),
    and save as .txt. Both the PDF and TXT are written to disk.
    """
    # --- fetch ---
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        pdf_bytes = resp.content
    except requests.RequestException as e:
        print(f"    ✗ Fetch error: {e}")
        return False

    # --- persist the raw PDF ---
    try:
        pdf_dest.write_bytes(pdf_bytes)
    except Exception as e:
        print(f"    ✗ PDF write error: {e}")
        return False

    # --- extract raw text from in-memory PDF ---
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages  = [page.extract_text() or "" for page in reader.pages]
        raw    = "\n".join(pages)
      #  raw = extract_text_no_headers(reader)
    except Exception as e:
        print(f"    ✗ PDF parse error: {e}")
        # PDF is already saved, so don't return False — just skip TXT
        print(f"    ⚠ PDF saved but TXT conversion failed — {pdf_dest.name}")
        return True

    # --- SCOTUS-specific cleaning ---
    body, footnotes = clean_scotus_text(raw)
    final_text      = assemble_output(body, footnotes)

    # --- write TXT ---
    try:
        txt_dest.write_text(final_text, encoding="utf-8")
    except Exception as e:
        print(f"    ✗ TXT write error: {e}")
        # PDF is already saved; partial failure
        print(f"    ⚠ PDF saved but TXT write failed — {pdf_dest.name}")
        return True

    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download SCOTUS opinion PDFs from CourtListener (2025–2026)."
    )
    parser.add_argument("--token",     required=True, help="CourtListener API token")
    parser.add_argument("--out-dir",   default="scotus_opinions", help="Output directory")
    parser.add_argument("--from-date", default="2025-01-01",  help="Start date YYYY-MM-DD")
    parser.add_argument("--to-date",   default="2025-01-31",  help="End date YYYY-MM-DD")
    parser.add_argument("--limit",     type=int, default=None, help="Max opinions to download")
    parser.add_argument("--dry-run",   action="store_true",    help="List only, no downloads")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(args.token)

    manifest_path  = out_dir / "manifest.csv"
    manifest_fields = [
        "opinion_id", "cluster_id", "docket_id", "case_name",
        "date_filed", "docket_number", "citations",
        "pdf_url", "pdf_file", "txt_file", "status",
    ]

    print(f"\n{'='*60}")
    print(f"  CourtListener SCOTUS Downloader")
    print(f"  Date range : {args.from_date} → {args.to_date}")
    print(f"  Output dir : {out_dir.resolve()}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'='*60}\n")

    downloaded = 0
    skipped    = 0
    no_pdf     = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=manifest_fields)
        writer.writeheader()

        print("Searching for SCOTUS opinions …\n")

        for opinion in search_opinions(session, args.from_date, args.to_date):
            if args.limit and (downloaded + skipped + no_pdf) >= args.limit:
                print(f"\nReached --limit of {args.limit}. Stopping.")
                break

            case_name  = opinion.get("caseName") or opinion.get("caseNameFull") or "Unknown"
            date_filed = opinion.get("dateFiled", "")
            docket_num = opinion.get("docketNumber", "")
            cluster_id = opinion.get("cluster_id", "")
            docket_id  = opinion.get("docket_id", "")
            citations  = "; ".join(opinion.get("citation", []))

            opinions_list = opinion.get("opinions", [{}])
            op            = opinions_list[0] if opinions_list else {}
            opinion_id    = op.get("id", cluster_id)

            pdf_url  = get_pdf_url(opinion)
            basename = safe_basename(case_name, opinion_id)
            pdf_dest = out_dir / f"{basename}.pdf"
            txt_dest = out_dir / f"{basename}.txt"

            row = {
                "opinion_id": opinion_id,
                "cluster_id": cluster_id,
                "docket_id": docket_id,
                "case_name": case_name,
                "date_filed": date_filed,
                "docket_number": docket_num,
                "citations": citations,
                "pdf_url": pdf_url or "",
                "pdf_file": f"{basename}.pdf" if pdf_url else "",
                "txt_file": f"{basename}.txt" if pdf_url else "",
                "status": "",
            }

            if not pdf_url:
                print(f"  — {case_name} ({date_filed}) … no PDF available")
                row["pdf_file"] = ""
                row["txt_file"] = ""
                row["status"] = "no_pdf"
                writer.writerow(row)
                csvfile.flush()
                no_pdf += 1
                continue

            if args.dry_run:
                print(f"  [DRY RUN] {case_name} ({date_filed})\n    → {pdf_url}")
                row["status"] = "dry_run"
                row["pdf_file"] = f"{basename}.pdf"
                row["txt_file"] = f"{basename}.txt"
                writer.writerow(row)
                csvfile.flush()
                continue

            if txt_dest.exists() and pdf_dest.exists():
                print(f"  ✓ {basename} (already exists, skipping)")
                row["status"] = "skipped"
                writer.writerow(row)
                csvfile.flush()
                skipped += 1
                continue

            print(f"  ↓ {case_name} ({date_filed})\n    → {basename}")
            success = fetch_and_convert(session, pdf_url, pdf_dest, txt_dest)

            if success:
                row["status"] = "downloaded"
                downloaded += 1
                pdf_size = pdf_dest.stat().st_size / 1024 if pdf_dest.exists() else 0
                txt_size = txt_dest.stat().st_size / 1024 if txt_dest.exists() else 0
                print(f"    ✓ saved — PDF {pdf_size:.1f} KB / TXT {txt_size:.1f} KB")
            else:
                row["status"] = "failed"
                print(f"    ✗ failed — check manifest for URL")

            writer.writerow(row)
            csvfile.flush()
            time.sleep(DOWNLOAD_DELAY)

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already existed)")
    print(f"  No PDF     : {no_pdf}")
    print(f"  Manifest   : {manifest_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
