from collections import Counter
from typing import List, Tuple

from ragenetics.privacy.vote import report_noisy_max
from ragenetics.privacy.accounting import Accountant
from ragenetics.privacy.sparse_vector import SVTGate


class DPSparseVoteRAG:
    """
    DP text generator that:
      • Proposes a baseline next token t0 (non-private).
      • Uses Sparse Vector Technique (SVT) to decide whether t0 is 'good enough'
        based on voter agreement rate (spends ε from SVT).
      • If SVT rejects, falls back to a DP noisy-max vote among voter proposals
        (spends ε_per_vote).
    """

    def __init__(
        self,
        voters: List,
        baseline_llm,
        epsilon_per_vote: float,
        svt: SVTGate,
        max_total_epsilon: float,
    ):
        """
        Args:
            voters: Objects with .agrees(question, prefix, candidate) -> bool and
                    .propose_next(question, prefix) -> str
            baseline_llm: Object with .sample_next_token(question, prefix, ctx) -> str
            epsilon_per_vote: ε spent when using noisy max on voter proposals
            svt: Sparse Vector Technique gate with .decide(score) -> (gate: bool, eps_used: float)
            max_total_epsilon: total ε budget for the whole generate() run
        """
        self.voters = voters
        self.baseline = baseline_llm
        self.eps_vote = float(epsilon_per_vote)
        self.svt = svt
        self.acc = Accountant(max_total_epsilon)

    def generate(self, question: str, max_tokens: int = 256) -> Tuple[str, float]:
        """
        Returns:
            (text, spent_epsilon)
        """
        out: List[str] = []
        if max_tokens <= 0:
            return "", float(getattr(self.acc, "spent", 0.0))

        stop_tokens = {"</s>", "<eos>", "\n"}  # extend as needed

        # Loop does not spend by itself; spending happens inside after decisions.
        while len(out) < max_tokens and self.acc.can_spend(0.0):
            prefix = " ".join(out)

            # 1) Non-private baseline suggestion
            t0 = self.baseline.sample_next_token(question, prefix=prefix, ctx=[])

            # 2) Private gate on agreement rate via SVT
            agreements = [int(v.agrees(question, prefix=prefix, candidate=t0)) for v in self.voters]
            denom = max(len(self.voters), 1)
            agree_rate = sum(agreements) / denom

            gate, eps_used = self.svt.decide(agree_rate)

            # Ensure we have budget for this SVT decision
            if not self.acc.can_spend(eps_used):
                break
            self.acc.spend(eps_used)

            if gate:
                # Accept baseline token
                last_tok = t0.strip()
                if last_tok:
                    out.append(last_tok)
                else:
                    # If t0 is empty, stop to avoid infinite loop
                    break
            else:
                # 3) Fall back to DP noisy-max vote
                if not self.acc.can_spend(self.eps_vote):
                    break

                props = [v.propose_next(question, prefix=prefix) for v in self.voters]
                props = [p.strip() for p in props if isinstance(p, str) and p.strip()]
                if not props:
                    break

                tok = report_noisy_max(Counter(props), epsilon=self.eps_vote)
                if not tok or not isinstance(tok, str):
                    break

                self.acc.spend(self.eps_vote)
                last_tok = tok.strip()
                out.append(last_tok)

            # 4) Stop on EOS token
            if last_tok in stop_tokens:
                break

        spent = float(getattr(self.acc, "spent", 0.0))
        return " ".join(out).strip(), spent
