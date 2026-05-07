import json
from pathlib import Path

from two_guards.core.writer import write_record


def test_write_record_passed(tmp_path: Path):
    record = {"plan": "A", "document_id": "doc_001", "liar": {"truth_flag": False}}

    write_record(
        output_dir=str(tmp_path),
        plan="plan_a",
        passed=True,
        record=record,
    )

    passed_dir = tmp_path / "plan_a" / "passed"
    assert passed_dir.exists()
    jsonl_files = list(passed_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["plan"] == "A"
    assert parsed["document_id"] == "doc_001"


def test_write_record_failed(tmp_path: Path):
    record = {"plan": "A", "document_id": "doc_002"}

    write_record(
        output_dir=str(tmp_path),
        plan="plan_a",
        passed=False,
        record=record,
    )

    failed_dir = tmp_path / "plan_a" / "failed"
    assert failed_dir.exists()
    jsonl_files = list(failed_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1


def test_write_record_appends(tmp_path: Path):
    for i in range(3):
        write_record(
            output_dir=str(tmp_path),
            plan="plan_b",
            passed=True,
            record={"index": i},
        )

    passed_dir = tmp_path / "plan_b" / "passed"
    jsonl_files = list(passed_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text().strip().split("\n")
    assert len(lines) == 3
