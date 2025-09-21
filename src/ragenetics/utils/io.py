import json
from pathlib import Path
from typing import Any, Dict


def load_bm25_index(path: Path) -> Dict[str, Any]:
    """
    Load a BM25 index from a JSON file.

    Args:
        path (Path): Path to the JSON file containing the serialized index.

    Returns:
        Dict[str, Any]: Parsed JSON object representing the index.
    """
    return json.loads(path.read_text(encoding="utf-8"))
