import json
from typing import Any, Dict, List, Sequence, Optional

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz


class LocalBM25Store:
    """
    Lightweight local BM25 document store with fuzzy de-duplication.

    Each doc is expected to be a mapping with at least a "text" field:
      {"id": "<optional-id>", "text": "<document text>"}
    """

    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []
        self.tokenized: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build(self, docs: List[Dict[str, Any]]) -> "LocalBM25Store":
        """
        Build (or rebuild) the BM25 index from docs.

        Args:
            docs: List of dicts with key "text".

        Returns:
            self
        """
        self.docs = docs or []
        self.tokenized = [d.get("text", "").lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None
        return self

    def similarity_search(self, query: str, k: int = 6) -> List[str]:
        """
        Retrieve up to k passages by BM25, then apply light fuzzy de-dup.

        Args:
            query: Query string.
            k: Max number of passages to return.

        Returns:
            List[str]: Top-k (approximately) unique passages.
        """
        if not self.docs or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query.lower().split())
        # Over-fetch then dedupe
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(1, k * 2)]

        picked: List[str] = []
        out: List[str] = []
        for i in order:
            txt = self.docs[i].get("text", "").strip()
            if not txt:
                continue
            # Keep if not ~duplicate of already picked (threshold 90)
            if all(fuzz.ratio(txt, p) < 90 for p in picked):
                picked.append(txt)
                out.append(txt)
                if len(out) >= k:
                    break
        return out

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize store to a JSON-serializable dict.
        """
        return {"docs": self.docs}

    @classmethod
    def deserialize(cls, obj: Dict[str, Any]) -> "LocalBM25Store":
        """
        Construct a store from a serialized dict produced by `serialize`.
        """
        store = cls()
        docs = obj.get("docs", [])
        if not isinstance(docs, list):
            docs = []
        return store.build(docs)
