from ragenetics.pipeline.dp_rag import DPVoteRAG


class DummyVoter:
    """
    Minimal dummy voter that always proposes the same token.
    """

    def __init__(self, token: str):
        self.tok = token

    def propose_next(self, q, prefix="") -> str:
        return self.tok


def test_dp_vote_spends_epsilon():
    """
    Sanity check: DPVoteRAG should consume ε when generating tokens.
    """
    voters = [DummyVoter("ok") for _ in range(3)]
    eng = DPVoteRAG(voters, epsilon_per_vote=0.5, delta=1e-6, max_total_epsilon=1.0)

    text, eps = eng.generate("q", max_tokens=3)

    assert isinstance(text, str)
    assert eps > 0, "No ε was spent — privacy accounting failed"
