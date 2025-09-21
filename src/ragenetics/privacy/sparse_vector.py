from .mechanisms import LaplaceMechanism


class SVTGate:
    """
    Sparse Vector Technique (SVT) gate for differentially-private decisions.

    Adds Laplace noise to the observed agreement score and checks if it
    exceeds the (noisy) threshold. Returns both the decision (True/False)
    and the ε spent for this gate check.
    """

    def __init__(self, threshold: float, epsilon_gate: float, epsilon_report: float):
        """
        Args:
            threshold (float): Agreement threshold for acceptance (0.0–1.0).
            epsilon_gate (float): ε used for the noisy threshold check.
            epsilon_report (float): ε budget reserved for reporting
                                    (not used in this simple implementation).
        """
        self.threshold = float(threshold)
        self.eps_gate = float(epsilon_gate)
        self.eps_rep = float(epsilon_report)

    def decide(self, agreement: float) -> tuple[bool, float]:
        """
        Decide whether to accept based on noisy agreement.

        Args:
            agreement (float): Observed agreement rate (0.0–1.0).

        Returns:
            (bool, float): Tuple of (decision, ε_spent_for_gate)
        """
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=self.eps_gate)
        noisy_agree = float(agreement) + float(mech.noise())
        return noisy_agree >= self.threshold, self.eps_gate
