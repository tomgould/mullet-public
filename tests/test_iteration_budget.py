"""Tests for IterationBudget."""

import threading
from mullet.core.agent.iteration_budget import IterationBudget


def test_basic_consume():
    budget = IterationBudget(max_total=3)
    assert budget.remaining == 3
    assert budget.consume() is True
    assert budget.remaining == 2
    assert budget.consume() is True
    assert budget.consume() is True
    assert budget.consume() is False
    assert budget.exhausted is True


def test_refund():
    budget = IterationBudget(max_total=5)
    budget.consume()
    budget.consume()
    assert budget.used == 2
    budget.refund(1)
    assert budget.used == 1
    assert budget.remaining == 4


def test_refund_below_zero():
    budget = IterationBudget(max_total=5)
    budget.refund(10)
    assert budget.used == 0


def test_thread_safety():
    budget = IterationBudget(max_total=100)
    successes = []

    def consume_many():
        count = 0
        for _ in range(200):
            if budget.consume():
                count += 1
        successes.append(count)

    threads = [threading.Thread(target=consume_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(successes) == 100
    assert budget.exhausted is True


def test_zero_budget():
    budget = IterationBudget(max_total=0)
    assert budget.consume() is False
    assert budget.exhausted is True


# --- Pressure thresholds ---

def test_pressure_none():
    budget = IterationBudget(max_total=100)
    assert budget.pressure == "none"
    assert budget.get_pressure_message() is None


def test_pressure_caution():
    budget = IterationBudget(max_total=10)
    for _ in range(7):  # 70% used
        budget.consume()
    assert budget.pressure == "caution"
    msg = budget.get_pressure_message()
    assert msg is not None
    assert "NOTE" in msg


def test_pressure_urgent():
    budget = IterationBudget(max_total=20)
    for _ in range(18):  # 90% used, 2 remaining (not critical)
        budget.consume()
    assert budget.pressure == "urgent"
    msg = budget.get_pressure_message()
    assert "URGENT" in msg


def test_pressure_critical():
    budget = IterationBudget(max_total=10)
    for _ in range(9):  # 1 remaining → critical
        budget.consume()
    assert budget.pressure == "critical"
    budget.consume()  # 0 remaining → still critical
    assert budget.pressure == "critical"
    msg = budget.get_pressure_message()
    assert "CRITICAL" in msg


def test_pressure_zero_budget():
    budget = IterationBudget(max_total=0)
    assert budget.pressure == "critical"


def test_timeout_blocks_consume():
    """Budget with expired timeout returns False on consume."""
    budget = IterationBudget(max_total=10, timeout_seconds=0.05)
    import time
    time.sleep(0.1)  # expire the timeout
    assert budget.consume() is False


def test_timeout_not_exceeded_allows_consume():
    """Budget with future timeout allows consume."""
    budget = IterationBudget(max_total=10, timeout_seconds=60)
    assert budget.consume() is True


def test_timed_out_property():
    """timed_out reflects wall-clock state."""
    budget = IterationBudget(max_total=10, timeout_seconds=0.05)
    import time
    assert budget.timed_out is False
    time.sleep(0.1)
    assert budget.timed_out is True


def test_no_timeout_never_times_out():
    """No timeout_seconds means timed_out is always False."""
    budget = IterationBudget(max_total=10)
    assert budget.timed_out is False


def test_elapsed_seconds_increases():
    """elapsed_seconds reflects real time."""
    import time
    budget = IterationBudget(max_total=10)
    time.sleep(0.05)
    assert budget.elapsed_seconds >= 0.04
