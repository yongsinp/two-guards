"""Document loader for plain-text legal source files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """A single source document loaded from disk.

    Attributes:
        id: Stem of the filename (e.g. "case_001" from "case_001.txt").
        text: Full text content of the document.
        source_path: Absolute path to the original file.
    """

    id: str
    text: str
    source_path: str


def load_documents(input_dir: str) -> list[Document]:
    """Load all .txt files from a directory as Document objects.

    Files are returned in sorted filename order. Non-.txt files are ignored.
    Returns an empty list if the directory does not exist.

    Args:
        input_dir: Path to the directory containing source .txt files.

    Returns:
        List of Document objects, one per .txt file found.
    """
    path = Path(input_dir)
    if not path.exists():
        return []

    documents = []
    for txt_file in sorted(path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=txt_file.stem,
                text=text,
                source_path=str(txt_file),
            )
        )
    return documents
