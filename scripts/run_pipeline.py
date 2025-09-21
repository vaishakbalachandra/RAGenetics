import argparse
import os
import json
import random
from pathlib import Path

import yaml
from loguru import logger

from ragenetics.retrieval.vectorstore import LocalBM25Store
from ragenetics.llm.local_openai import build_llm
from ragenetics.pipeline.dp_rag import DPVoteRAG
from ragenetics.pipeline.dp_sparse_rag import DPSparseVoteRAG
from ragenetics.privacy.sparse_vector import SVTGate
from ragenetics.utils.io import load_bm25_index
from ragenetics.llm.base import VoterLLM

# Set deterministic random seed for reproducibility
random.seed(7)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run privacy-preserving RAG pipeline.")
    ap.add_argument("--config", required=True, help="Path to YAML configuration file")
    ap.add_argument("--query", required=True, help="Query string to run")
    ap.add_argument("--log", default="runs/last_run.jsonl", help="Path to JSONL log file")
    args = ap.parse_args()

    # Load configuration
    cfg = yaml.safe_load(open(args.config))

    # Ensure log directory exists
    os.makedirs(Path(args.log).parent, exist_ok=True)

    # Load BM25 vector store
    idx_path = Path("data/embeddings/bm25_index.json")
    if not idx_path.exists():
        logger.warning("No vector index at data/embeddings/bm25_index.json; build it using scripts/build_vectorstore.py")
        store = LocalBM25Store()  # empty store
    else:
        store = LocalBM25Store.deserialize(load_bm25_index(idx_path))

    # Build voters and baseline LLM
    cfg_llm = cfg.get("llm") or {"provider": "mock", "model": "debug-mock", "max_tokens": 256}
    llm = build_llm(cfg_llm)

    voters = [VoterLLM(store, llm) for _ in range(cfg["privacy"]["m_voters"])]

    # Select privacy scheme
    if cfg["privacy"]["scheme"] == "dp_vote":
        engine = DPVoteRAG(
            voters,
            cfg["privacy"]["epsilon_per_vote"],
            cfg["privacy"]["delta"],
            cfg["privacy"]["max_total_epsilon"],
        )
    else:
        gate = SVTGate(
            cfg["privacy"]["svt"]["threshold"],
            cfg["privacy"]["svt"]["epsilon_gate"],
            cfg["privacy"]["svt"]["epsilon_report"],
        )
        engine = DPSparseVoteRAG(
            voters,
            llm,
            cfg["privacy"]["epsilon_per_vote"],
            gate,
            cfg["privacy"]["max_total_epsilon"],
        )

    # Generate answer under DP constraints
    text, eps = engine.generate(args.query, max_tokens=cfg["llm"].get("max_tokens", 256))

    print(f"ε spent: {round(eps, 3)}")
    print("\n=== ANSWER ===\n", text)

    # Append run log entry
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": args.query, "eps_spent": eps}) + "\n")
