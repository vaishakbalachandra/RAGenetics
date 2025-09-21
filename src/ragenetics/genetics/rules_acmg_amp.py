# Non-diagnostic illustrative templates

def evidence_sentences(gene: str, variant: str):
    """
    Generate illustrative evidence sentences for a given gene and variant.

    Args:
        gene (str): Gene symbol (e.g., BRCA1).
        variant (str): Variant notation (e.g., c.123A>G).

    Returns:
        List[str]: List of ACMG-like evidence strings (for demonstration).
    """
    return [
        f"PM1: {gene} variant {variant} lies in a critical domain (if applicable).",
        f"PP3: Multiple lines of computational evidence support a deleterious effect (demo).",
        f"PM3: In trans with a pathogenic variant reported in recessive cases (if literature supports).",
    ]
