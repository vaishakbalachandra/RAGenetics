from collections import Counter
from typing import List, Tuple

from ragenetics.privacy.vote import report_noisy_max
from ragenetics.privacy.accounting import Accountant


class DPVoteRAG:
    """
    Differentially-private next-token generator using a committee of voter LLMs.
    Each step:
      1) Ask all voters to propose a next token.
      2) Aggregate with noisy max (ε per step).
      3) Spend ε; stop when max_tokens reached, budget exhausted, or EOS token seen.
    """

    def __init__(self, voters: List, epsilon_per_vote: float, delta: float, max_total_epsilon: float):
        """
        Args:
            voters: List of voter objects, each with `propose_next(question, prefix) -> str`.
            epsilon_per_vote: ε spent per noisy max step.
            delta: δ for DP accounting (kept for compatibility if Accountant uses it elsewhere).
            max_total_epsilon: total ε budget available.
        """
        self.voters = voters
        self.eps_vote = float(epsilon_per_vote)
        self.delta = float(delta)
        self.acc = Accountant(max_total_epsilon)

    def generate(self, question: str, max_tokens: int = 256) -> Tuple[str, float]:
        """
        Generate text under a DP budget.

        Args:
            question: The user query.
            max_tokens: Maximum number of tokens to emit.

        Returns:
            (text, spent_epsilon)
        """
        out: List[str] = []
        if not self.voters:
            return "", getattr(self.acc, "spent", 0.0)

        stop_tokens = {"</s>", "<eos>", "\n"}  # extend as needed

        while len(out) < max_tokens and self.acc.can_spend(self.eps_vote):
            prefix = " ".join(out)

            # Collect proposals; skip empty strings to avoid degenerate votes
            props = [v.propose_next(question, prefix=prefix) for v in self.voters]
            props = [p for p in props if isinstance(p, str) and p.strip()]

            # If no voter produced a token, stop early
            if not props:
                break

            tok = report_noisy_max(Counter(props), epsilon=self.eps_vote)

            # Defensive fallback if the voting returns an empty/None token
            if not tok or not isinstance(tok, str):
                break

            out.append(tok)
            self.acc.spend(self.eps_vote)

            if tok in stop_tokens:
                break

        # Some Accountant implementations track `spent` as an attribute or property
        spent = getattr(self.acc, "spent", 0.0)
        return " ".join(out).strip(), float(spent)
