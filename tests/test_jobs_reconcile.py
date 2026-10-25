"""Checking the ledger's spend against what each provider says it billed."""

from mullet.jobs.reconcile import Reconciliation, provider_spent, reconcile, take


def test_a_provider_that_reports_usage_gives_its_rise():
    before = {"used_usd": 50.0, "remaining_usd": None}
    after = {"used_usd": 58.5, "remaining_usd": None}
    assert provider_spent(before, after) == 8.5


def test_a_provider_that_reports_a_balance_gives_its_fall():
    assert provider_spent({"used_usd": None, "remaining_usd": 100.0}, {"used_usd": None, "remaining_usd": 70.5}) == 29.5


def test_a_missing_or_partial_snapshot_gives_none():
    assert provider_spent(None, {"used_usd": 1.0}) is None
    assert provider_spent({"used_usd": None, "remaining_usd": None}, {"used_usd": None, "remaining_usd": None}) is None


def test_a_topped_up_balance_is_never_negative_spend():
    assert provider_spent({"remaining_usd": 10.0}, {"remaining_usd": 60.0}) == 0.0


def test_close_figures_agree():
    found = Reconciliation("kimi", recorded=29.94, provider_says=30.06)
    assert not found.off and "agrees" in found.line()


def test_a_large_shortfall_in_the_ledger_is_flagged_and_says_why_it_may_not_be_a_fault():
    found = Reconciliation("openrouter", recorded=8.04, provider_says=18.04)
    assert found.off
    line = found.line()
    assert "$10.00 more than recorded" in line and "Anything else using the same key" in line


def test_a_tiny_job_is_not_flagged_for_a_few_cents():
    assert not Reconciliation("kimi", recorded=0.05, provider_says=0.20).off


def test_reconcile_pairs_each_provider_and_notes_a_missing_figure():
    before = {"openrouter": {"used_usd": 20.0}, "kimi": {"remaining_usd": 100.0}}
    after = {"openrouter": {"used_usd": 25.0}}
    found = {r.provider: r for r in reconcile(before, after, {"openrouter": 4.9, "kimi": 1.0})}
    assert found["openrouter"].provider_says == 5.0 and not found["openrouter"].off
    assert found["kimi"].provider_says is None and "wasn't available" in found["kimi"].line()


def test_a_failing_probe_gives_an_empty_snapshot():
    def broken():
        raise RuntimeError("offline")

    assert take(broken) == {}
