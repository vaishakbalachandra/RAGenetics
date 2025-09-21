import hashlib


def sha1(text: str) -> str:
    """
    Compute the SHA-1 hash of a string and return its hex digest.

    Args:
        text (str): Input string.

    Returns:
        str: SHA-1 hex digest.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
