from typing import List
from ragenetics.retrieval.rankers import heuristic_boost


class VoterLLM:
    """
    A wrapper around a retriever + language model that can propose
    next tokens and vote on candidate completions.
    """

    def __init__(self, retriever, model):
        """
        Initialize the VoterLLM.

        Args:
            retriever: Object that supports similarity_search(question, k).
            model: Object that supports sample_next_token() and yesno().
        """
        self.retriever = retriever
        self.model = model

    def propose_next(self, question: str, prefix: str = "") -> str:
        """
        Propose the next token or continuation for a given question.

        Args:
            question (str): User question or query.
            prefix (str): Existing partial completion.

        Returns:
            str: Proposed next token or text.
        """
        ctx = self.retriever.similarity_search(question, k=6)
        ctx = heuristic_boost(question, ctx)
        return self.model.sample_next_token(question, prefix, ctx)

    def agrees(self, question: str, prefix: str, candidate: str) -> bool:
        """
        Evaluate whether the model agrees with a candidate answer.

        Args:
            question (str): User question or query.
            prefix (str): Partial context text.
            candidate (str): Candidate answer to validate.

        Returns:
            bool: True if model agrees, False otherwise.
        """
        ctx = self.retriever.similarity_search(question, k=6)
        return self.model.yesno(question, prefix, candidate, ctx)
