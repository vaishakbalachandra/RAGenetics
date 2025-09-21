class Accountant:
    """
    Simple privacy budget accountant.
    Tracks total ε spent and ensures it does not exceed the max budget.
    """

    def __init__(self, max_total: float):
        """
        Args:
            max_total (float): Maximum ε budget allowed.
        """
        self.max_total = float(max_total)
        self.spent = 0.0

    def can_spend(self, eps: float) -> bool:
        """
        Check if there is enough budget left to spend ε.

        Args:
            eps (float): ε to check.

        Returns:
            bool: True if spending is allowed, False otherwise.
        """
        return self.spent + eps <= self.max_total

    def spend(self, eps: float):
        """
        Deduct ε from remaining budget.

        Args:
            eps (float): ε to deduct.
        """
        self.spent += float(eps)
