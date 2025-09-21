from pathlib import Path
from typing import List, Dict


def read_and_chunk_dir(root: Path, chunk_size: int = 1200, overlap: int = 100) -> List[Dict[str, str]]:
    """
    Recursively read all .txt and .md files under `root`, chunk them into
    overlapping pieces, and return a list of {"id": ..., "text": ...} dicts.

    Args:
        root (Path): Root directory to search.
        chunk_size (int): Number of characters per chunk.
        overlap (int): Number of characters to overlap between chunks.

    Returns:
        List[Dict[str, str]]: List of chunks with IDs for reference.
    """
    docs: List[Dict[str, str]] = []

    for p in sorted(root.glob("**/*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            i = 0
            while i < len(text):
                docs.append({"id": f"{p.name}:{i}", "text": text[i:i + chunk_size]})
                i += max(1, chunk_size - overlap)

    return docs
