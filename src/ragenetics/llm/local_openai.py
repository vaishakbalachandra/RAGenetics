import os
import random
from typing import List, Optional


class MockLLM:
    """
    Tiny mock LLM for testing. Samples a 'next token' from the question/context bag.
    """

    def __init__(self, seed: int = 7):
        random.seed(seed)
        self.vocab = [
            ",",
            ".",
            "and",
            "with",
            "the",
            "is",
            "suggests",
            "phenotype",
            "variant",
            "likely",
            "pathogenic",
        ]

    def sample_next_token(self, question: str, prefix: str, ctx: List[str]) -> str:
        """
        Extremely simple heuristic: prefer words from question/context; else fallback vocab.
        """
        bag: List[str] = []
        bag += question.lower().split()
        for c in ctx:
            bag += [w for w in c.lower().split() if len(w) > 3]
        if bag:
            return random.choice(bag[:50])
        return random.choice(self.vocab)

    def yesno(self, question: str, prefix: str, candidate: str, ctx: List[str]) -> bool:
        """
        'Agree' when the candidate token appears in any context or prefix (case-insensitive).
        """
        text = " ".join(ctx).lower() + " " + prefix.lower()
        return candidate.strip().lower() in text


class OpenAILLM:
    """
    OpenAI-backed LLM wrapper.

    Expects environment variables by default:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL (optional)
      - OPENAI_MODEL (optional; defaults to 'gpt-4o-mini')
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def sample_next_token(self, question: str, prefix: str, ctx: List[str]) -> str:
        """
        Ask the model to emit just the next token.
        """
        prompt = (
            f"Question: {question}\n"
            f"Context:\n"
            f"{'\n'.join(ctx)}\n"
            f"Given the partial answer: '{prefix}', emit just the next token."
        )
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
        )
        content = (r.choices[0].message.content or "").strip()
        # Return the first whitespace-separated token if present; else empty string
        return content.split()[0] if content else ""

    def yesno(self, question: str, prefix: str, candidate: str, ctx: List[str]) -> bool:
        """
        Ask the model to answer yes/no on whether the next token equals `candidate`.
        """
        prompt = (
            f"Question: {question}\n"
            f"Context:\n"
            f"{'\n'.join(ctx)}\n"
            f"Given partial answer '{prefix}', is the next token exactly '{candidate}'? Reply yes or no."
        )
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
        )
        reply = (r.choices[0].message.content or "").lower()
        return "yes" in reply


def build_llm(cfg: dict):
    """
    Factory to build an LLM from config.
    cfg keys:
      - provider: "openai" | "mock" (default: "mock")
      - model: str (OpenAI model name, optional)
      - base_url: str (optional)
      - api_key: str (optional)
    """
    provider = cfg.get("provider", "mock").lower()
    if provider == "openai":
        return OpenAILLM(
            model=cfg.get("model"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
        )
    return MockLLM()
