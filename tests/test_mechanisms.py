import numpy as np
from ragenetics.privacy.mechanisms import LaplaceMechanism


def test_laplace_noise_variance():
    """
    Sanity check for LaplaceMechanism:
    - Mean of generated noise should be close to 0
    - Variance should be roughly 2 * (sensitivity / epsilon)^2
    """
    mech = LaplaceMechanism(1.0, 0.5)
    x = mech.noise(size=10000)

    # Mean should be near zero
    assert abs(np.mean(x)) < 0.2

    # (Optional) Variance check - expected variance for Laplace(b) is 2 * b^2
    b = mech.sens / mech.eps
    expected_var = 2 * b**2
    observed_var = np.var(x)
    assert abs(observed_var - expected_var) / expected_var < 0.25  # within 25%
