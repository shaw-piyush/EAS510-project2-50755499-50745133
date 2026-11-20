"""Routing metrics: quality-vs-cost curve, AUC, cost@95%, classifier metrics.

Implements PRD Section 7 exactly. Every function works on ONE split given as
equal-length numpy arrays; averaging over seeds is done by Piyush's runner.

Notation (one entry per prompt):
    scores   router score s_i, higher = safer to send to the cheap model
             (the router's estimate of P(cheap model is correct))
    q_cheap  cheap model's correctness score   (qc_i)
    q_strong strong model's correctness score  (qs_i)
    c_cheap  cheap model's actual cost         (cc_i)
    c_strong strong model's actual cost        (cs_i)

Routing rule used everywhere: the prompts with the LOWEST scores go to the
strong model, because the router is least sure the cheap model can answer
them. A threshold t means "send to cheap if s >= t".
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config
from src.policies import _as_arrays

# Comparisons against the 95% target allow this much floating-point slack.
# Quality(n) is built from a cumulative sum, so it can differ from
# mean(q_strong) in the last bits; without slack a curve could "miss" a
# target it reaches exactly.
_EPS = 1e-12


def routing_curve(scores, q_cheap, q_strong, c_cheap, c_strong):
    """Quality-vs-cost curve of a router (PRD 7.1 to 7.3).

    Sort prompts by score ascending; for k = 0..n send the k lowest-scored
    prompts to strong and the rest to cheap. Quality(k) and Cost(k) are means
    over all n prompts of the chosen model's score and cost.

    Computed with cumulative sums, not a loop over k: switching prompt i from
    cheap to strong changes the total quality by (qs_i - qc_i) and the total
    cost by (cs_i - cc_i), so Quality(k) = (sum(qc) + cumsum(qs - qc)[k]) / n.

    Ties (PRD 7.2): a threshold cannot split prompts with equal scores, so we
    evaluate only at the boundaries between blocks of equal scores. Joining
    those points with straight lines is the expected curve under random
    tie-breaking, and it does not depend on the order of tied rows (the
    cumulative sum at the end of a block is the same whatever order the block
    is in). This matters for kNN and k-means, whose scores repeat.

    Returns a DataFrame with columns:
        frac_strong  k / n
        cost         Cost(k)
        quality      Quality(k)
        x            normalized cost: 0 = always-cheap, 1 = always-strong
    The first row is always-cheap (k = 0), the last is always-strong (k = n).
    """
    scores, q_cheap, q_strong, c_cheap, c_strong = _as_arrays(
        scores, q_cheap, q_strong, c_cheap, c_strong
    )
    n = len(scores)

    # Stable sort so the result is deterministic; ties are handled below by
    # only reading the curve at block boundaries, so tie order cannot matter.
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    dq = (q_strong - q_cheap)[order]
    dc = (c_strong - c_cheap)[order]
    cum_q = np.concatenate([[0.0], np.cumsum(dq)])
    cum_c = np.concatenate([[0.0], np.cumsum(dc)])

    # Block boundaries: k = 0, every k where the score changes, and k = n.
    change = np.flatnonzero(s_sorted[1:] != s_sorted[:-1]) + 1
    ks = np.concatenate([[0], change, [n]])

    quality = (q_cheap.sum() + cum_q[ks]) / n
    cost = (c_cheap.sum() + cum_c[ks]) / n

    cost_cheap, cost_strong = cost[0], cost[-1]
    if cost_strong == cost_cheap:
        # x would be 0/0. This only happens if both models cost the same on
        # this split, in which case routing has nothing to trade off.
        raise ValueError("always-cheap and always-strong have the same cost; "
                         "normalized cost is undefined")
    x = (cost - cost_cheap) / (cost_strong - cost_cheap)

    return pd.DataFrame({
        "frac_strong": ks / n,
        "cost": cost,
        "quality": quality,
        "x": x,
    })


def curve_auc(curve):
    """Area under Quality over normalized cost x, from 0 to 1 (PRD 7.4).

    Trapezoid rule in the order of k (the row order of routing_curve). We do
    not re-sort by x: per-prompt costs can make x step backwards when the
    strong model happens to be cheaper on a prompt, and the PRD defines the
    area in the order of k.
    """
    return float(np.trapezoid(curve["quality"].to_numpy(), curve["x"].to_numpy()))


def random_auc(q_cheap_mean, q_strong_mean):
    """AUC of random routing: (Q_cheap + Q_strong) / 2 (PRD 7.4).

    Random routing's expected curve is the straight line from (0, Q_cheap)
    to (1, Q_strong), and the area under a straight line on [0, 1] is the
    mean of its ends. Lift = curve_auc(router) - random_auc(...).
    """
    return (float(q_cheap_mean) + float(q_strong_mean)) / 2.0


def cost_at_95_test(curve, q_strong_mean, c_strong_mean,
                    target=config.TARGET_QUALITY_FRAC):
    """Cost@95%, read off the test curve (PRD 7.5).

    Target = target * always-strong test quality. Returns the smallest
    Cost(k) whose Quality(k) reaches the target, as a percentage of
    always-strong cost.

    Always defined: k = n is always-strong itself, which reaches 95% of its
    own quality (for non-negative scores). This is an optimistic "best case"
    number because it peeks at test; the honest version is the
    validation-chosen threshold (choose_threshold + apply_threshold).
    """
    goal = target * float(q_strong_mean) - _EPS
    reached = curve["quality"].to_numpy() >= goal
    if not reached.any():
        # Cannot happen for curves from routing_curve; guard anyway.
        raise ValueError("no point on the curve reaches the target")
    best_cost = curve["cost"].to_numpy()[reached].min()
    return float(100.0 * best_cost / float(c_strong_mean))


def choose_threshold(scores_val, q_cheap_val, q_strong_val, c_cheap_val, c_strong_val,
                     target=config.TARGET_QUALITY_FRAC):
    """Pick the threshold t* on VALIDATION (PRD 7.6).

    Among all thresholds a router can actually use (one per block boundary
    of the validation curve), return the one that reaches
    target * always-strong validation quality with the least validation cost.
    The rule is then "send to cheap if s >= t*".

    Threshold for boundary k: the score of the first prompt that goes to
    cheap, i.e. the (k+1)-th smallest score. For k = n (everything to strong)
    t* = +inf, so no score passes it.

    Fallback: if no cheaper point reaches the target, k = n always does, so
    t* = +inf ("send everything to strong"). The function therefore always
    returns a threshold. If two points cost the same, the higher-quality one
    wins.
    """
    s, qc, qs, cc, cs = _as_arrays(scores_val, q_cheap_val, q_strong_val,
                                   c_cheap_val, c_strong_val)
    curve = routing_curve(s, qc, qs, cc, cs)
    n = len(s)
    s_sorted = np.sort(s, kind="stable")
    ks = np.rint(curve["frac_strong"].to_numpy() * n).astype(int)

    goal = target * qs.mean() - _EPS
    quality = curve["quality"].to_numpy()
    cost = curve["cost"].to_numpy()
    ok = np.flatnonzero(quality >= goal)
    # Least cost first, then highest quality (lexsort uses the last key first).
    best = ok[np.lexsort((-quality[ok], cost[ok]))[0]]

    k = ks[best]
    return float(np.inf) if k == n else float(s_sorted[k])


def apply_threshold(scores, t, q_cheap, q_strong, c_cheap, c_strong,
                    target=config.TARGET_QUALITY_FRAC):
    """Apply a validation-chosen threshold to TEST (PRD 7.6).

    Sends a prompt to cheap if s >= t, else to strong. Returns:
        cost_pct    test cost as a percentage of always-strong test cost
        quality     test quality (mean correctness of the chosen model)
        met_target  whether test quality reached target * always-strong
                    test quality
    """
    s, qc, qs, cc, cs = _as_arrays(scores, q_cheap, q_strong, c_cheap, c_strong)
    to_cheap = s >= t
    cost = np.where(to_cheap, cc, cs).mean()
    quality = np.where(to_cheap, qc, qs).mean()
    return {
        "cost_pct": float(100.0 * cost / cs.mean()),
        "quality": float(quality),
        "met_target": bool(quality >= target * qs.mean() - _EPS),
    }


def classifier_metrics(y_true, scores, t):
    """ROC-AUC, precision, recall and F1 of the router as a classifier (PRD 7.7).

    Positive class = the cheap model is correct (y = 1). ROC-AUC uses the raw
    scores, so it does not depend on t. Precision, recall and F1 use the
    decision "predict cheap-correct if s >= t", with t the validation-chosen
    t*. Precision then reads: of the prompts sent to cheap, the share it got
    right.

    Undefined values are returned as NaN instead of a misleading 0:
    ROC-AUC when the test split has only one class, and precision (and F1)
    when no prompt is sent to cheap (e.g. t* = +inf). The runner should
    aggregate with nanmean and report how many seeds were NaN.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    s, = _as_arrays(scores)
    if len(y_true) != len(s):
        raise ValueError("y_true and scores have different lengths")
    y_pred = (s >= t).astype(int)

    roc = float(roc_auc_score(y_true, s)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "roc_auc": roc,
        "precision": float(precision_score(y_true, y_pred, zero_division=np.nan)),
        "recall": float(recall_score(y_true, y_pred, zero_division=np.nan)),
        "f1": float(f1_score(y_true, y_pred, zero_division=np.nan)),
    }
