# !/usr/bin/env python3
"""
CourtListener SCOTUS Opinion Downloader (2025–2026)
====================================================
Downloads Supreme Court opinion PDFs filed between 2025-01-01 and 2026-12-31.

Usage:
    python downloader.py --token YOUR_API_TOKEN [options]

Options:
    --token     API token from courtlistener.com/profile/  (required)
    --out-dir   Directory to save PDFs (default: ./scotus_opinions)
    --from-date Start date YYYY-MM-DD (default: 2025-01-01)
    --to-date   End date   YYYY-MM-DD (default: 2026-12-31)
    --limit     Max number of opinions to download (default: unlimited)
    --dry-run   List cases without downloading files
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://www.courtlistener.com"
SEARCH_URL = f"{BASE_URL}/api/rest/v4/search/"
STORAGE_URL = "https://storage.courtlistener.com"

# Polite delay between PDF downloads (seconds) — respect rate limits
DOWNLOAD_DELAY = 0.5


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
        "type": "o",  # opinions
        "court": "scotus",  # Supreme Court of the United States
        "filed_after": from_date,
        "filed_before": to_date,
        "order_by": "dateFiled asc",
        "page_size": 20,
    }

    url = SEARCH_URL
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

        data = resp.json()
        results = data.get("results", [])
        print(f" {len(results)} results (total found: {data.get('count', '?')})")

        yield from results

        # Cursor-based pagination: follow the 'next' URL verbatim
        url = data.get("next")
        params = None  # params are already embedded in 'next' URL


def get_pdf_url(opinion: dict) -> str | None:
    """
    Extract the best available PDF URL from a search result opinion entry.
    Falls back through: local_path → download_url → None
    """
    for op in opinion.get("opinions", []):
        local = op.get("local_path")
        if local:
            # local_path is a relative path like "2025/01/15/foo.pdf"
            return f"{STORAGE_URL}/{local}"

        download = op.get("download_url")
        if download and download.lower().endswith(".pdf"):
            return download

    return None


def safe_filename(case_name: str, opinion_id: int) -> str:
    """Build a safe filename from the case name and opinion ID."""
    clean = "".join(c if c.isalnum() or c in " -_" else "_" for c in case_name)
    clean = clean.strip().replace("  ", " ")[:80]
    return f"{opinion_id}_{clean}.pdf"


def download_pdf(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single PDF. Returns True on success."""
    try:
        resp = session.get(url, timeout=30, stream=True)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"    ✗ Download error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download SCOTUS opinion PDFs from CourtListener (2025–2026)."
    )
    parser.add_argument("--token", required=True, help="CourtListener API token")
    parser.add_argument("--out-dir", default="scotus_opinions", help="Output directory")
    parser.add_argument("--from-date", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--to-date", default="2025-01-31", help="End date YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Max opinions to download")
    parser.add_argument("--dry-run", action="store_true", help="List only, no downloads")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(args.token)

    # CSV manifest — always written even in dry-run mode
    manifest_path = out_dir / "manifest.csv"
    manifest_fields = [
        "opinion_id", "cluster_id", "docket_id", "case_name",
        "date_filed", "docket_number", "citations",
        "pdf_url", "local_file", "status",
    ]

    print(f"\n{'=' * 60}")
    print(f"  CourtListener SCOTUS Downloader")
    print(f"  Date range : {args.from_date} → {args.to_date}")
    print(f"  Output dir : {out_dir.resolve()}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'=' * 60}\n")

    downloaded = 0
    skipped = 0
    no_pdf = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=manifest_fields)
        writer.writeheader()

        print("Searching for SCOTUS opinions …\n")

        for opinion in search_opinions(session, args.from_date, args.to_date):
            if args.limit and (downloaded + skipped + no_pdf) >= args.limit:
                print(f"\nReached --limit of {args.limit}. Stopping.")
                break

            case_name = opinion.get("caseName") or opinion.get("caseNameFull") or "Unknown"
            date_filed = opinion.get("dateFiled", "")
            docket_num = opinion.get("docketNumber", "")
            cluster_id = opinion.get("cluster_id", "")
            docket_id = opinion.get("docket_id", "")
            citations = "; ".join(opinion.get("citation", []))

            # Each search result can have multiple opinion entries (majority, dissent, etc.)
            opinions_list = opinion.get("opinions", [{}])
            op = opinions_list[0] if opinions_list else {}
            opinion_id = op.get("id", cluster_id)

            pdf_url = get_pdf_url(opinion)
            filename = safe_filename(case_name, opinion_id)
            dest = out_dir / filename

            row = {
                "opinion_id": opinion_id,
                "cluster_id": cluster_id,
                "docket_id": docket_id,
                "case_name": case_name,
                "date_filed": date_filed,
                "docket_number": docket_num,
                "citations": citations,
                "pdf_url": pdf_url or "",
                "local_file": filename if pdf_url else "",
                "status": "",
            }

            if not pdf_url:
                print(f"  — {case_name} ({date_filed}) … no PDF available")
                row["status"] = "no_pdf"
                writer.writerow(row)
                csvfile.flush()
                no_pdf += 1
                continue

            if args.dry_run:
                print(f"  [DRY RUN] {case_name} ({date_filed})\n    → {pdf_url}")
                row["status"] = "dry_run"
                writer.writerow(row)
                csvfile.flush()
                continue

            if dest.exists():
                print(f"  ✓ {filename} (already exists, skipping)")
                row["status"] = "skipped"
                writer.writerow(row)
                csvfile.flush()
                skipped += 1
                continue

            print(f"  ↓ {case_name} ({date_filed})\n    → {filename}")
            success = download_pdf(session, pdf_url, dest)

            if success:
                row["status"] = "downloaded"
                downloaded += 1
                print(f"    ✓ saved ({dest.stat().st_size / 1024:.1f} KB)")
            else:
                row["status"] = "failed"
                print(f"    ✗ failed — check manifest for URL")

            writer.writerow(row)
            csvfile.flush()
            time.sleep(DOWNLOAD_DELAY)

    print(f"\n{'=' * 60}")
    print(f"  Done.")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already existed)")
    print(f"  No PDF     : {no_pdf}")
    print(f"  Manifest   : {manifest_path.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
