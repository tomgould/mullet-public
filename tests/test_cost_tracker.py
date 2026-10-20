import pytest
from mullet.core.providers.cost_tracker import CostTracker
from mullet.core.providers.model_info import MODEL_DATABASE, get_pricing

def test_cost_tracker_init():
    tracker = CostTracker()
    assert tracker._records == []

def test_record_single_entry():
    tracker = CostTracker()
    model = "gpt-4o"
    tokens_in = 100
    tokens_out = 50
    
    input_cost_per_1k, output_cost_per_1k = get_pricing(model)
    expected_cost = (tokens_in / 1000 * input_cost_per_1k) + \
                    (tokens_out / 1000 * output_cost_per_1k)

    tracker.record(model, tokens_in, tokens_out)
    assert len(tracker._records) == 1
    record = tracker._records[0]
    assert record["model"] == model
    assert record["tokens_in"] == tokens_in
    assert record["tokens_out"] == tokens_out
    assert record["cost"] == expected_cost
    # assert record["provider"] == provider # Provider information is not directly available from get_pricing

def test_record_multiple_entries():
    tracker = CostTracker()
    tracker.record("gpt-4o", 100, 50)
    tracker.record("gpt-4o-mini", 200, 100)
    assert len(tracker._records) == 2

def test_record_invalid_model():
    tracker = CostTracker()
    # The CostTracker no longer raises an error for unknown models as get_pricing returns default 0.0 costs.
    tracker.record("unknown-model", 10, 10)
    assert len(tracker._records) == 1
    assert tracker._records[0]["cost"] == 0.0

def test_total_cost():
    tracker = CostTracker()
    gpt4_input, gpt4_output = get_pricing("gpt-4o")
    gpt35_input, gpt35_output = get_pricing("gpt-4o-mini") # Using gpt-4o-mini as gpt-3.5-turbo is not in the MODEL_DATABASE directly

    tracker.record("gpt-4o", 100, 50)
    tracker.record("gpt-4o-mini", 200, 100)
    expected_total = (100 / 1000 * gpt4_input) + (50 / 1000 * gpt4_output) + \
                     (200 / 1000 * gpt35_input) + (100 / 1000 * gpt35_output)
    assert tracker.total_cost() == pytest.approx(expected_total)

def test_breakdown_by_model():
    tracker = CostTracker()
    gpt4_input, gpt4_output = get_pricing("gpt-4o")
    gpt4_cost = (300 / 1000 * gpt4_input) + (150 / 1000 * gpt4_output)
    gpt35_input, gpt35_output = get_pricing("gpt-4o-mini")
    gpt35_cost = (50 / 1000 * gpt35_input) + (25 / 1000 * gpt35_output)

    tracker.record("gpt-4o", 100, 50)
    tracker.record("gpt-4o", 200, 100)
    tracker.record("gpt-4o-mini", 50, 25)

    breakdown = tracker.breakdown_by_model()
    assert breakdown["gpt-4o"] == pytest.approx(gpt4_cost)
    assert breakdown["gpt-4o-mini"] == pytest.approx(gpt35_cost)
    assert sum(breakdown.values()) == pytest.approx(tracker.total_cost())

def test_empty_cost_tracker():
    tracker = CostTracker()
    assert tracker.total_cost() == 0.0
    assert tracker.breakdown_by_model() == {}

def test_record_zero_tokens():
    tracker = CostTracker()
    tracker.record("gpt-4o", 0, 0)
    assert len(tracker._records) == 1
    assert tracker._records[0]["cost"] == 0.0
    assert tracker.total_cost() == 0.0

def test_get_pricing_function():
    input_cost, output_cost = get_pricing("gpt-4o")
    assert input_cost == MODEL_DATABASE["gpt-4o"].input_cost_per_1k
    assert output_cost == MODEL_DATABASE["gpt-4o"].output_cost_per_1k

    # For an unknown model, get_pricing returns default 0.0 costs.
    input_cost, output_cost = get_pricing("unknown-model")
    assert input_cost == 0.0
    assert output_cost == 0.0


def test_record_negative_tokens_raises():
    tracker = CostTracker()
    with pytest.raises(ValueError):
        tracker.record("some-model", tokens_in=-1, tokens_out=100)


def test_record_unknown_model_raises(monkeypatch):
    tracker = CostTracker()
    monkeypatch.setattr(
        "mullet.core.providers.cost_tracker.get_pricing",
        lambda model: (None, None),
    )
    with pytest.raises(ValueError):
        tracker.record("totally-unknown-model-xyz", tokens_in=100, tokens_out=100)
