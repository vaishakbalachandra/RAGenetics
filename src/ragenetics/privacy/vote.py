import numpy as np
from .mechanisms import LaplaceMechanism


def report_noisy_max(counts: dict[str, int], epsilon: float) -> str:
    """
    Differentially private noisy argmax.

    Adds Laplace noise to each count and returns the key with the highest
    noisy value.

    Args:
        counts (dict[str, int]): Dictionary mapping items to counts.
        epsilon (float): Privacy parameter ε.

    Returns:
        str: Key corresponding to the noisy maximum. Empty string if counts is empty.
    """
    if not counts:
        return ""

    keys = list(counts.keys())
    arr = np.array([counts[k] for k in keys], dtype=float)

    # Laplace noise with L1 sensitivity = 1
    mech = LaplaceMechanism(sensitivity=1.0, epsilon=epsilon)
    noisy = arr + mech.noise(size=arr.shape)

    return keys[int(np.argmax(noisy))]
