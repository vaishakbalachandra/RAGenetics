import re
from typing import List


HPO_HINTS = ["HP:", "phenotype", "syndrome", "dysmorph", "seizure", "short stature"]
GENE_HINTS = ["CFTR", "BRCA", "FBN1", "PAH", "PKD1", "COL1A1"]


def heuristic_boost(query: str, passages: List[str]) -> List[str]:
    """
    Re-rank passages by boosting those containing phenotype or gene hints.

    Args:
        query (str): User query string.
        passages (list[str]): Retrieved text passages.

    Returns:
        list[str]: Passages reordered so those with hints are prioritized.
    """
    q = query.lower()

    # Count hint matches in the query
    hints = 0
    hints += any(h.lower() in q for h in HPO_HINTS)
    hints += any(g.lower() in q for g in GENE_HINTS)

    # If query has no hints, return as-is
    if not hints:
        return passages

    # Score passages: boost those that mention HPO or gene hints
    scored = []
    for p in passages:
        score = 0
        for h in HPO_HINTS + GENE_HINTS:
            if re.search(re.escape(h), p, re.I):
                score += 1
        scored.append((score, p))

    # Sort by score (descending), then by length (shorter first)
    scored.sort(key=lambda x: (-x[0], len(x[1])))

    return [p for _, p in scored]
