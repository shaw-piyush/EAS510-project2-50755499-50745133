"""Tests for src/metrics.py: every required test from PRD Section 7, plus a few
checks on the threshold functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable; see the note in tests/test_policies.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (  # noqa: E402
    apply_threshold,
    choose_threshold,
    classifier_metrics,
    cost_at_95_test,
    curve_auc,
    random_auc,
    routing_curve,
)
from src.policies import reference_policies  # noqa: E402


def make_data(n=400, seed=0, qc_rate=0.5, qs_rate=0.85):
    """Synthetic split: binary correctness, strong pricier on every prompt."""
    rng = np.random.default_rng(seed)
    qc = (rng.random(n) < qc_rate).astype(float)
    qs = (rng.random(n) < qs_rate).astype(float)
    cc = rng.uniform(0.1, 0.5, n)
    cs = cc + rng.uniform(1.0, 3.0, n)
    return qc, qs, cc, cs


# ---------------------------------------------------------------- required


def test_random_routing_auc_equals_mean_of_endpoints():
    # A router that gives every prompt the same score cannot rank them, so it
    # IS random routing; its curve is the straight line between the ends.
    qc, qs, cc, cs = make_data()
    curve = routing_curve(np.zeros(len(qc)), qc, qs, cc, cs)
    assert abs(curve_auc(curve) - random_auc(qc.mean(), qs.mean())) < 1e-9


def test_curve_runs_from_always_cheap_to_always_strong():
    qc, qs, cc, cs = make_data()
    scores = np.random.default_rng(1).random(len(qc))
    curve = routing_curve(scores, qc, qs, cc, cs)
    first, last = curve.iloc[0], curve.iloc[-1]
    assert first["x"] == 0.0 and first["frac_strong"] == 0.0
    assert last["x"] == 1.0 and last["frac_strong"] == 1.0
    assert first["quality"] == pytest.approx(qc.mean())
    assert last["quality"] == pytest.approx(qs.mean())
    assert first["cost"] == pytest.approx(cc.mean())
    assert last["cost"] == pytest.approx(cs.mean())


def test_shuffling_tied_rows_gives_same_curve_and_auc():
    qc, qs, cc, cs = make_data()
    rng = np.random.default_rng(2)
    # Only 6 distinct score values, so there are many ties (like kNN / k-means).
    scores = rng.integers(0, 6, len(qc)) / 5.0
    base = routing_curve(scores, qc, qs, cc, cs)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(qc))
        shuffled = routing_curve(scores[perm], qc[perm], qs[perm], cc[perm], cs[perm])
        assert np.allclose(base.to_numpy(), shuffled.to_numpy())
        assert curve_auc(shuffled) == pytest.approx(curve_auc(base), abs=1e-12)
    # Only block boundaries are kept: 6 blocks -> 7 points.
    assert len(base) == len(np.unique(scores)) + 1


@pytest.mark.parametrize("seed", range(5))
def test_oracle_quality_at_least_any_router_point(seed):
    qc, qs, cc, cs = make_data(seed=seed)
    oracle_q = reference_policies(qc, qs, cc, cs).loc["oracle", "quality"]
    rng = np.random.default_rng(seed + 100)
    noisy = qc + rng.normal(0, 0.7, len(qc))  # an imperfect but informative router
    for scores in (rng.random(len(qc)), noisy, qc):
        curve = routing_curve(scores, qc, qs, cc, cs)
        assert oracle_q >= curve["quality"].max() - 1e-12


def test_perfect_router_reaches_oracle_quality():
    qc, qs, cc, cs = make_data()
    oracle_q = reference_policies(qc, qs, cc, cs).loc["oracle", "quality"]
    y = qc  # score = true label (cheap correct)
    curve = routing_curve(y, qc, qs, cc, cs)
    assert curve["quality"].max() == pytest.approx(oracle_q, abs=1e-12)


def test_cost_at_95_random_like_is_100_percent():
    # Built so cheap quality is far below 95% of strong: an all-tied router
    # only has the two endpoints, so only always-strong reaches the target.
    qc, qs, cc, cs = make_data(qc_rate=0.3, qs_rate=0.9)
    curve = routing_curve(np.zeros(len(qc)), qc, qs, cc, cs)
    assert cost_at_95_test(curve, qs.mean(), cs.mean()) == pytest.approx(100.0)


def test_cost_at_95_perfect_router_below_100_percent():
    qc, qs, cc, cs = make_data(qc_rate=0.3, qs_rate=0.9)
    curve = routing_curve(qc, qc, qs, cc, cs)
    assert cost_at_95_test(curve, qs.mean(), cs.mean()) < 100.0


# ---------------------------------------------------------------- extras


def test_auc_of_perfect_router_beats_random():
    qc, qs, cc, cs = make_data()
    curve = routing_curve(qc, qc, qs, cc, cs)
    assert curve_auc(curve) > random_auc(qc.mean(), qs.mean())


def test_choose_threshold_meets_target_on_validation():
    qc, qs, cc, cs = make_data(seed=3)
    rng = np.random.default_rng(3)
    scores = qc + rng.normal(0, 0.5, len(qc))
    t = choose_threshold(scores, qc, qs, cc, cs)
    res = apply_threshold(scores, t, qc, qs, cc, cs)
    assert res["met_target"]
    # Applying t* on the same split must give the curve's cheapest
    # target-reaching cost, i.e. the test-read number on that split.
    curve = routing_curve(scores, qc, qs, cc, cs)
    assert res["cost_pct"] == pytest.approx(cost_at_95_test(curve, qs.mean(), cs.mean()))


def test_choose_threshold_falls_back_to_all_strong():
    # All-tied router on data where cheap is far below target: only
    # "everything to strong" works, so t* must be +inf.
    qc, qs, cc, cs = make_data(qc_rate=0.3, qs_rate=0.9)
    t = choose_threshold(np.zeros(len(qc)), qc, qs, cc, cs)
    assert t == np.inf
    res = apply_threshold(np.zeros(len(qc)), t, qc, qs, cc, cs)
    assert res["cost_pct"] == pytest.approx(100.0)
    assert res["met_target"]


def test_threshold_respects_ties():
    # Tied prompts must go the same way: t* is one of the actual scores (or inf).
    qc, qs, cc, cs = make_data(seed=4)
    scores = np.random.default_rng(4).integers(0, 4, len(qc)).astype(float)
    t = choose_threshold(scores, qc, qs, cc, cs)
    assert t == np.inf or t in set(scores)


def test_classifier_metrics_values():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    m = classifier_metrics(y, s, t=0.5)
    # predicted cheap-correct: rows 0 and 2 -> TP=1, FP=1, FN=1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_classifier_metrics_undefined_cases_are_nan():
    m = classifier_metrics([1, 1, 1], [0.2, 0.5, 0.9], t=np.inf)
    assert np.isnan(m["roc_auc"])      # one class only
    assert np.isnan(m["precision"])    # nothing sent to cheap


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        routing_curve([0.1, 0.2], [1, 0], [1, 1], [0.1, 0.1], [1.0])
