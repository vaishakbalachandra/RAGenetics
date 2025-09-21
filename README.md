# RAGenetics – Privacy‑Preserving RAG over Genetic Test Reports


RAGenetics implements two differentially‑private RAG algorithms:


- **DPVoteRAG** — sample‑and‑aggregate: multiple voters read disjoint shards; release each token via **Report‑Noisy‑Max**.
- **DPSparseVoteRAG** — adds a **Sparse Vector Technique (SVT)** gate: only spend ε when voters disagree with a non‑RAG baseline token.


> ⚠️ Research reference only. No clinical use. Ships **synthetic** data generator.


## Quickstart
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env # set OPENAI_API_KEY or leave empty to use a mock LLM
python scripts/build_vectorstore.py --data data/toy_reports --out data/embeddings
python scripts/run_pipeline.py --config configs/dp_small.yaml --query "Which HPO terms suggest a ciliopathy?"
python scripts/run_pipeline.py --config configs/dp_sparse.yaml --query "Summarize evidence for CFTR p.Phe508del"