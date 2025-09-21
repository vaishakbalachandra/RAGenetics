import re

# Regex to capture HGVS-like variant notations (c., g., p.)
HGVS_G = re.compile(r"[cgp]\.[A-Za-z0-9_>+\-()]+")


def find_hgvs(text: str) -> list[str]:
    """
    Find HGVS-like variant strings in a given text.

    Args:
        text (str): The input clinical text.

    Returns:
        list[str]: List of all HGVS-like variant matches.
    """
    return HGVS_G.findall(text)
