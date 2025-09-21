from ragenetics.privacy.sparse_vector import SVTGate


def test_svt_gate_monotonic():
    """
    Basic sanity check:
    Higher agreement should be at least as likely to pass the gate as lower agreement.
    """
    g = SVTGate(threshold=0.5, epsilon_gate=0.5, epsilon_report=0.5)

    ok_low, _ = g.decide(0.1)
    ok_high, _ = g.decide(0.9)

    # If low agreement passes, high agreement must also pass
    assert (not ok_low) or ok_high
