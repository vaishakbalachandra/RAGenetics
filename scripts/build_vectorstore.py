import argparse
import os
import json
from pathlib import Path

from ragenetics.retrieval.vectorstore import LocalBM25Store
from ragenetics.retrieval.chunking import read_and_chunk_dir


if __name__ == "__main__":
    # Parse CLI arguments
    ap = argparse.ArgumentParser(description="Build a BM25 index from a directory of text/markdown files.")
    ap.add_argument("--data", required=True, help="Path to directory containing .txt/.md files")
    ap.add_argument("--out", required=True, help="Output directory where index will be saved")
    args = ap.parse_args()

    # Read and chunk documents
    docs = read_and_chunk_dir(Path(args.data))

    # Build BM25 index
    store = LocalBM25Store()
    store.build(docs)

    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)

    # Write serialized index to disk
    idx_path = Path(args.out) / "bm25_index.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(store.serialize(), f, indent=2)

    print(f"Wrote {idx_path}")
