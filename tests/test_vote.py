from ragenetics.privacy.vote import report_noisy_max


def test_noisy_max_returns_key():
    """
    Sanity check: report_noisy_max should always return one of the keys.
    """
    tok = report_noisy_max({"a": 3, "b": 1}, epsilon=1.0)
    assert tok in {"a", "b"}
