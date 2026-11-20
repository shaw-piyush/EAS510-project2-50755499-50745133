"""Tests for src/policies.py (PRD 8.1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable (for `config` and `src.*`) when pytest is run
# as plain `pytest` from the repo root, without needing a shared conftest.py
# or pytest config file (those are repo-wide files, not Avimta's).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policies import (  # noqa: E402
    oracle_choice,
    random_curve,
    reference_policies,
    simulate_random,
)


def make_data(n=500, seed=0):
    """Synthetic two-model split: binary scores, strong better and pricier.

    Costs are drawn so the strong model costs more on every prompt, which is
    the situation the "oracle cost <= always-strong cost" guarantee assumes.
    """
    rng = np.random.default_rng(seed)
    qc = (rng.random(n) < 0.55).astype(float)
    qs = (rng.random(n) < 0.80).astype(float)
    cc = rng.uniform(0.1, 0.5, n)
    cs = cc + rng.uniform(1.0, 3.0, n)
    return qc, qs, cc, cs


@pytest.mark.parametrize("seed", range(5))
def test_oracle_never_worse_than_either_model(seed):
    qc, qs, cc, cs = make_data(seed=seed)
    ref = reference_policies(qc, qs, cc, cs)
    assert ref.loc["oracle", "quality"] >= ref.loc["always_cheap", "quality"]
    assert ref.loc["oracle", "quality"] >= ref.loc["always_strong", "quality"]


@pytest.mark.parametrize("seed", range(5))
def test_oracle_cost_never_above_always_strong(seed):
    qc, qs, cc, cs = make_data(seed=seed)
    ref = reference_policies(qc, qs, cc, cs)
    assert ref.loc["oracle", "cost"] <= ref.loc["always_strong", "cost"]


def test_oracle_rule_matches_prd():
    # cheap right -> cheap; cheap wrong & strong right -> strong; both wrong -> cheap
    qc = np.array([1, 1, 0, 0])
    qs = np.array([1, 0, 1, 0])
    assert oracle_choice(qc, qs).tolist() == [False, False, True, False]


def test_reference_policies_shape_and_endpoints():
    qc, qs, cc, cs = make_data()
    ref = reference_policies(qc, qs, cc, cs)
    assert list(ref.index) == ["always_cheap", "always_strong", "random", "oracle"]
    assert {"cost", "quality"} <= set(ref.columns)
    assert ref.loc["always_cheap", "x"] == pytest.approx(0.0)
    assert ref.loc["always_strong", "x"] == pytest.approx(1.0)
    assert ref.loc["always_cheap", "quality"] == pytest.approx(qc.mean())
    assert ref.loc["always_strong", "cost"] == pytest.approx(cs.mean())


def test_random_curve_is_straight_line_between_endpoints():
    qc, qs, cc, cs = make_data()
    rc = random_curve(qc, qs, cc, cs)
    assert rc["quality"].iloc[0] == pytest.approx(qc.mean())
    assert rc["quality"].iloc[-1] == pytest.approx(qs.mean())
    # Constant slope = straight line.
    slopes = np.diff(rc["quality"]) / np.diff(rc["x"])
    assert np.allclose(slopes, slopes[0])


def test_simulated_random_matches_expectation():
    qc, qs, cc, cs = make_data(n=2000)
    draws = np.array([simulate_random(qc, qs, cc, cs, p=0.5, seed=s) for s in range(200)])
    assert draws[:, 0].mean() == pytest.approx((cc.mean() + cs.mean()) / 2, rel=0.01)
    assert draws[:, 1].mean() == pytest.approx((qc.mean() + qs.mean()) / 2, rel=0.01)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        reference_policies([1, 0], [1, 1, 0], [0.1, 0.1], [1.0, 1.0])
