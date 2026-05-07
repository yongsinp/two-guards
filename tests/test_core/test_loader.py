from pathlib import Path

from two_guards.core.loader import Document, load_documents


def test_load_documents_from_directory(tmp_path: Path):
    (tmp_path / "case_001.txt").write_text("This is a legal case about contracts.")
    (tmp_path / "case_002.txt").write_text("This is a case about torts.")
    (tmp_path / "notes.md").write_text("This should be ignored.")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    texts = {d.text for d in docs}
    assert "This is a legal case about contracts." in texts
    assert "This is a case about torts." in texts


def test_load_documents_sets_id_and_source(tmp_path: Path):
    (tmp_path / "statute.txt").write_text("Section 1. Definitions.")

    docs = load_documents(str(tmp_path))

    assert docs[0].id == "statute"
    assert docs[0].source_path == str(tmp_path / "statute.txt")


def test_load_documents_empty_dir(tmp_path: Path):
    docs = load_documents(str(tmp_path))
    assert docs == []
