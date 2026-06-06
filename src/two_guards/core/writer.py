"""JSONL dataset writer with passed/failed routing."""

import json
from pathlib import Path

_SESSION_FILE_CACHE: dict[str, Path] = {}


def write_record(
    output_dir: str,
    plan: str,
    passed: bool | None,
    record: dict,
    split: str | None = None,
) -> None:
    """Append a dataset record as a JSON line to the appropriate output file.

    Records are written to ``<output_dir>/<plan>/<split>/``. By default,
    split is derived from ``passed`` as ``passed``/``failed``.
    All records for a given plan/split are appended to a stable JSONL file.

    Args:
        output_dir: Root output directory (e.g. "data").
        plan: Sub-directory name for the plan (e.g. "plan_a").
        passed: True/False quality signal used when split is not explicitly set.
        record: Dict to serialise as a JSON line.
        split: Optional explicit split name (e.g. "original").
    """
    if split is None:
        if passed is None:
            raise ValueError("Either split must be provided or passed must be boolean")
        subdir = "passed" if passed else "failed"
    else:
        subdir = split

    dir_path = Path(output_dir) / plan / subdir
    dir_path.mkdir(parents=True, exist_ok=True)

    cache_key = f"{output_dir}/{plan}/{subdir}"
    if cache_key not in _SESSION_FILE_CACHE:
        filename = f"{plan}.jsonl"
        _SESSION_FILE_CACHE[cache_key] = dir_path / filename

    file_path = _SESSION_FILE_CACHE[cache_key]
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
