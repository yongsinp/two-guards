"""JSONL dataset writer with passed/failed routing."""

import json
from pathlib import Path

_SESSION_FILE_CACHE: dict[str, Path] = {}


def write_record(output_dir: str, plan: str, passed: bool, record: dict) -> None:
    """Append a dataset record as a JSON line to the appropriate output file.

    Records are written to ``<output_dir>/<plan>/passed/`` or
    ``<output_dir>/<plan>/failed/`` depending on the quality signal.
    All records from the same session are appended to the same timestamped
    file so runs produce one file per plan/split rather than one per record.

    Args:
        output_dir: Root output directory (e.g. "data").
        plan: Sub-directory name for the plan (e.g. "plan_a").
        passed: True if the record is high-quality (hallucination survived
            detection); False if it was caught.
        record: Dict to serialise as a JSON line.
    """
    subdir = "passed" if passed else "failed"
    dir_path = Path(output_dir) / plan / subdir
    dir_path.mkdir(parents=True, exist_ok=True)

    cache_key = f"{output_dir}/{plan}/{subdir}"
    if cache_key not in _SESSION_FILE_CACHE:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{plan}_{timestamp}.jsonl"
        _SESSION_FILE_CACHE[cache_key] = dir_path / filename

    file_path = _SESSION_FILE_CACHE[cache_key]
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
