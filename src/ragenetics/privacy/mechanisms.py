from dataclasses import dataclass
import numpy as np
import math


@dataclass
class DPParams:
    """
    Dataclass for Differential Privacy parameters.
    """
    epsilon: float
    delta: float = 0.0


class LaplaceMechanism:
    """
    Implements the Laplace mechanism for differential privacy.
    """

    def __init__(self, sensitivity: float, epsilon: float):
        """
        Args:
            sensitivity (float): L1 sensitivity of the query.
            epsilon (float): Privacy parameter ε.
        """
        self.sens = float(sensitivity)
        # Avoid divide-by-zero
        self.eps = max(float(epsilon), 1e-12)

    def noise(self, size=None):
        """
        Generate Laplace noise for a given sensitivity and ε.

        Args:
            size (int | tuple | None): Shape of noise array. None = scalar.

        Returns:
            np.ndarray | float: Laplace-distributed noise.
        """
        scale = self.sens / self.eps
        return np.random.laplace(0.0, scale, size=size)


class GaussianMechanism:
    """
    Implements the Gaussian mechanism for (ε, δ)-Differential Privacy.
    """

    def __init__(self, sensitivity: float, epsilon: float, delta: float):
        """
        Args:
            sensitivity (float): L2 sensitivity of the query.
            epsilon (float): Privacy parameter ε.
            delta (float): Privacy parameter δ.
        """
        self.sens = float(sensitivity)
        self.eps = max(float(epsilon), 1e-12)
        self.delta = max(float(delta), 1e-12)

    def sigma(self) -> float:
        """
        Compute the Gaussian standard deviation σ.

        Returns:
            float: σ value for noise calibration.
        """
        return (self.sens * math.sqrt(2.0 * math.log(1.25 / self.delta))) / self.eps

    def noise(self, size=None):
        """
        Generate Gaussian noise for a given sensitivity, ε, and δ.

        Args:
            size (int | tuple | None): Shape of noise array. None = scalar.

        Returns:
            np.ndarray | float: Gaussian-distributed noise.
        """
        return np.random.normal(0.0, self.sigma(), size=size)
