"""Reference policies for two-model routing (PRD 8.1).

These are the fixed points every router is compared against:

- always_cheap / always_strong: the two ends of every quality-vs-cost curve.
- random: send each prompt to strong with probability p. Its *expected*
  quality and cost are linear in p, so the expected curve is the straight
  line between the two endpoints. A router must beat this line to be useful.
- oracle: knows the answers (uses test labels by definition), so it is an
  upper reference point, never a router (PRD 0.4).

Everything here is a pure function of per-prompt numpy arrays, so it needs
nothing from the data pipeline: the caller passes the cheap and strong
models' scores and costs for one split. Which models are "cheap" and
"strong" is decided by D2 in config.py, not here.

The multi-model versions (always-cheapest by mean *training* cost, and the
multi-model oracle) belong to the extension and are added with
src/multimodel.py (PRD 8.6).
"""

import numpy as np
import pandas as pd

import config


def _as_arrays(*arrays):
    """Convert inputs to 1-D float arrays and check they line up.

    Every policy and metric compares prompts row by row, so a length mismatch
    would silently pair the wrong prompts; fail loudly instead.
    """
    out = [np.asarray(a, dtype=float).ravel() for a in arrays]
    n = len(out[0])
    if n == 0:
        raise ValueError("inputs are empty")
    if any(len(a) != n for a in out):
        raise ValueError(f"inputs have different lengths: {[len(a) for a in out]}")
    if any(np.isnan(a).any() for a in out):
        raise ValueError("inputs contain NaN")
    return out


def oracle_choice(q_cheap, q_strong):
    """Per-prompt oracle decision: True = send to strong.

    Rule (PRD 8.1): cheap if cheap is correct; else strong if strong is
    correct; else cheap (both wrong, so take the cheaper one).

    Implemented as "strong only when it scores strictly higher than cheap".
    With 0/1 scores (D1) this is exactly the PRD rule. Writing it this way
    also does the sensible thing if a score is not 0/1: pick the better
    model, and on a tie pick cheap, i.e. "the cheapest model with the best
    score", the same rule the multi-model oracle uses.
    """
    q_cheap, q_strong = _as_arrays(q_cheap, q_strong)
    return q_strong > q_cheap


def random_curve(q_cheap, q_strong, c_cheap, c_strong, n_points=11):
    """Expected curve of random routing as p goes from 0 to 1.

    Expected quality = (1 - p) * mean(q_cheap) + p * mean(q_strong), and the
    same for cost, so the curve is the straight line between always-cheap and
    always-strong. We return the exact expectation rather than a simulation
    so the reference line has no sampling noise.
    """
    q_cheap, q_strong, c_cheap, c_strong = _as_arrays(q_cheap, q_strong, c_cheap, c_strong)
    p = np.linspace(0.0, 1.0, n_points)
    qc, qs = q_cheap.mean(), q_strong.mean()
    cc, cs = c_cheap.mean(), c_strong.mean()
    return pd.DataFrame({
        "frac_strong": p,
        "cost": (1 - p) * cc + p * cs,
        "quality": (1 - p) * qc + p * qs,
        "x": p,  # normalized cost of the expected curve equals p exactly
    })


def simulate_random(q_cheap, q_strong, c_cheap, c_strong, p=0.5, seed=None):
    """One seeded draw of random routing, as a check on random_curve.

    Optional per PRD 8.1. Returns (cost, quality) of a single random routing
    where each prompt goes to strong with probability p. Averaged over many
    seeds it should approach the expected line.
    """
    q_cheap, q_strong, c_cheap, c_strong = _as_arrays(q_cheap, q_strong, c_cheap, c_strong)
    # Seed comes from config so a run is reproducible from config.py alone.
    rng = np.random.default_rng(config.SEEDS[0] if seed is None else seed)
    to_strong = rng.random(len(q_cheap)) < p
    cost = np.where(to_strong, c_strong, c_cheap).mean()
    quality = np.where(to_strong, q_strong, q_cheap).mean()
    return float(cost), float(quality)


def reference_policies(q_cheap, q_strong, c_cheap, c_strong):
    """Cost and quality of the four reference policies on one split.

    Returns a DataFrame indexed by policy name
    (always_cheap, always_strong, random, oracle) with columns
    cost and quality (the agreed interface, PRD 5.5), plus frac_strong and
    x (normalized cost, 0 = always-cheap, 1 = always-strong) so plots can
    place the markers on the same axis as the router curves.

    The "random" row is the expected point at p = 0.5, i.e. the midpoint of
    the random line; use random_curve() for the whole line.
    """
    q_cheap, q_strong, c_cheap, c_strong = _as_arrays(q_cheap, q_strong, c_cheap, c_strong)
    cc, cs = c_cheap.mean(), c_strong.mean()
    qc, qs = q_cheap.mean(), q_strong.mean()

    to_strong = oracle_choice(q_cheap, q_strong)
    oracle_cost = np.where(to_strong, c_strong, c_cheap).mean()
    oracle_quality = np.where(to_strong, q_strong, q_cheap).mean()

    rows = {
        "always_cheap": (cc, qc, 0.0),
        "always_strong": (cs, qs, 1.0),
        "random": ((cc + cs) / 2, (qc + qs) / 2, 0.5),
        "oracle": (oracle_cost, oracle_quality, to_strong.mean()),
    }
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["cost", "quality", "frac_strong"])
    # Same normalization as metrics.routing_curve. If the two models cost the
    # same on this split, normalized cost is undefined; leave it NaN rather
    # than divide by zero.
    span = cs - cc
    df["x"] = (df["cost"] - cc) / span if span != 0 else np.nan
    return df
