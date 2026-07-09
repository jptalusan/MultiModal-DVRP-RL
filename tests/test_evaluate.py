"""Evaluation-harness tests. Marked ``network`` — builds real MOSAIC envs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dvrp_rl.evaluate import evaluate, run_episode
from dvrp_rl.policies import AcceptAll, RandomPolicy

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "binghampton.yaml"
SEEDS = [100, 101]
STEPS = 120


@pytest.mark.network
def test_evaluate_reports_sane_aggregates():
    res = evaluate(CONFIG, lambda _s: AcceptAll(), SEEDS, n_steps=STEPS)
    assert len(res["episodes"]) == len(SEEDS)
    assert 0.0 <= res["mean_service_rate"] <= 1.0
    assert all(e["total_requests"] > 0 for e in res["episodes"])


@pytest.mark.network
def test_same_seed_gives_identical_demand_totals():
    """Seed fixes the demand stream: AcceptAll and Random face the same
    requests, so total_requests matches for a given seed."""
    a = run_episode(CONFIG, AcceptAll(), seed=100, n_steps=STEPS)
    r = run_episode(CONFIG, RandomPolicy(accept_prob=0.5, seed=100), seed=100, n_steps=STEPS)
    assert a["total_requests"] == r["total_requests"]


@pytest.mark.network
def test_accept_all_beats_random_on_service_rate():
    """Random rejects ~half outright (counted as unserved), so AcceptAll —
    the baseline to beat — should have a clearly higher service rate."""
    accept = evaluate(CONFIG, lambda _s: AcceptAll(), SEEDS, n_steps=STEPS)
    rand = evaluate(CONFIG, lambda s: RandomPolicy(accept_prob=0.5, seed=s), SEEDS, n_steps=STEPS)
    assert accept["mean_service_rate"] > rand["mean_service_rate"]
