"""Data check: the evidence for decisions D1-D11 (PRD 6.3, 9 Tue Nov 17).

    python scripts/data_check.py

Reads the official pickle from data/raw/ and writes CSVs to
results/full/data_check/. It decides nothing: every table is evidence for a
decision Piyush and Avimta make together.

Outputs hold counts and statistics only, never prompt or response text, so
they are safe to commit even before we know whether the data may be shared.

Caveat on D2: the per-model means here are over the WHOLE dataset. The final
cheap/strong pair must be confirmed on the training split (PRD 0.4).
"""

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402

COST_SUFFIX = "|total_cost"
RESPONSE_SUFFIX = "|model_response"


# --------------------------------------------------------------------------
# Column roles, inferred from the real schema rather than hard-coded
# --------------------------------------------------------------------------
def column_roles(df):
    """Map each column to a role. Model names come from the cost columns."""
    models = [c[: -len(COST_SUFFIX)] for c in df.columns if c.endswith(COST_SUFFIX)]
    roles = {}
    for c in df.columns:
        if c.endswith(COST_SUFFIX):
            roles[c] = "cost"
        elif c.endswith(RESPONSE_SUFFIX):
            roles[c] = "response"
        elif c in models:
            roles[c] = "score"
        elif c == "prompt":
            roles[c] = "prompt"
        elif c == "eval_name":
            roles[c] = "benchmark (never a feature)"
        elif c == "sample_id":
            roles[c] = "id (contains benchmark name; never a feature)"
        elif c.startswith("oracle"):
            roles[c] = "derived from scores (never a feature)"
        else:
            roles[c] = "other"
    return models, roles


def candidate_group(eval_name):
    """A *candidate* parent-benchmark mapping for D3 discussion, not a decision.

    Rule: mmlu-* -> mmlu, mtbench* -> mtbench, chinese* (any case) -> chinese,
    everything else stays as its own group.
    """
    if eval_name.startswith("mmlu-"):
        return "mmlu"
    if eval_name.startswith("mtbench"):
        return "mtbench"
    if eval_name.lower().startswith("chinese"):
        return "chinese"
    return eval_name


def parse_prompt(raw):
    """Prompts are stored as str(list_of_turns). Return the list, or None.

    ast.literal_eval only evaluates Python literals, so this is safe on
    untrusted text (unlike eval).
    """
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, list) else None


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------
def schema_table(df, roles):
    rows = []
    for c in df.columns:
        s = df[c]
        types = s.map(lambda v: type(v).__name__).value_counts()
        n_empty = int((s.map(lambda v: isinstance(v, str) and v.strip() == "")).sum())
        rows.append(
            {
                "column": c,
                "role": roles[c],
                "dtype": str(s.dtype),
                "python_types": "; ".join(f"{k}:{v}" for k, v in types.items()),
                "n_missing": int(s.isna().sum()),
                "n_empty_string": n_empty,
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def eval_counts(df):
    t = df["eval_name"].value_counts().rename_axis("eval_name").reset_index(name="n_rows")
    t["candidate_group"] = t["eval_name"].map(candidate_group)
    return t.sort_values("eval_name").reset_index(drop=True)


def group_counts(df):
    g = df.assign(candidate_group=df["eval_name"].map(candidate_group))
    t = g.groupby("candidate_group").agg(
        n_eval_names=("eval_name", "nunique"), n_rows=("eval_name", "size")
    )
    return t.sort_values("n_rows", ascending=False).reset_index()


def score_values(df, models):
    """Long table: unique score values per model per eval_name (D1)."""
    rows = []
    for e, sub in df.groupby("eval_name"):
        for m in models:
            vals = sorted(pd.unique(sub[m].dropna().astype(float)))
            rows.append(
                {
                    "eval_name": e,
                    "model": m,
                    "n_rows": len(sub),
                    "n_unique": len(vals),
                    "values": " ".join(f"{v:g}" for v in vals),
                    "is_binary": set(vals) <= {0.0, 1.0},
                }
            )
    return pd.DataFrame(rows)


def binary_by_eval(sv, df, models):
    """One row per eval_name: is it 0/1 for every model, and if not, how far off? (D1)

    `share_scores_not_0_or_1` pools all (row, model) scores in the eval: it
    says whether a non-binary eval is "mostly binary with a few odd values"
    or genuinely graded, which changes what D1 should do with it.
    """
    rows = []
    for e, sub in sv.groupby("eval_name"):
        all_vals = sorted({float(v) for s in sub["values"] for v in s.split()})
        pooled = pd.Series(df.loc[df["eval_name"] == e, models].astype(float).to_numpy().ravel())
        counts = pooled.value_counts().sort_index()
        rows.append(
            {
                "eval_name": e,
                "candidate_group": candidate_group(e),
                "n_rows": int(sub["n_rows"].iloc[0]),
                "all_models_binary": bool(sub["is_binary"].all()),
                "n_models_nonbinary": int((~sub["is_binary"]).sum()),
                "distinct_values_all_models": " ".join(f"{v:g}" for v in all_vals),
                "n_scores_pooled": len(pooled),
                "share_scores_not_0_or_1": float((~pooled.isin([0.0, 1.0])).mean()),
                "pooled_value_counts": " ".join(f"{k:g}:{v}" for k, v in counts.items()),
            }
        )
    return pd.DataFrame(rows)


def model_summary(df, models, binary_evals):
    """Per-model mean score and cost over all rows (D2 evidence; see caveat)."""
    bin_mask = df["eval_name"].isin(binary_evals)
    rows = []
    for m in models:
        cost = df[m + COST_SUFFIX]
        score = df[m].astype(float)
        rows.append(
            {
                "model": m,
                "mean_score_all_rows": score.mean(),
                "mean_score_binary_evals_only": score[bin_mask].mean(),
                "mean_cost": cost.mean(),
                "median_cost": cost.median(),
                "total_cost": cost.sum(),
                "min_cost": cost.min(),
                "max_cost": cost.max(),
                "n_zero_cost": int((cost == 0).sum()),
                "n_negative_cost": int((cost < 0).sum()),
                "n_missing_cost": int(cost.isna().sum()),
                "n_missing_score": int(df[m].isna().sum()),
            }
        )
    t = pd.DataFrame(rows).sort_values("mean_score_binary_evals_only", ascending=False)
    best = t["mean_score_binary_evals_only"].max()
    best_cost = t.loc[t["mean_score_binary_evals_only"].idxmax(), "mean_cost"]
    t["gap_to_best_binary"] = best - t["mean_score_binary_evals_only"]
    t["cost_ratio_to_best"] = t["mean_cost"] / best_cost
    return t.reset_index(drop=True)


def by_group_wide(df, models, kind):
    """Wide table: candidate group x model, mean score or mean cost."""
    g = df.assign(candidate_group=df["eval_name"].map(candidate_group))
    cols = models if kind == "score" else [m + COST_SUFFIX for m in models]
    t = g.groupby("candidate_group")[cols].agg(lambda s: s.astype(float).mean())
    t.columns = models
    t.insert(0, "n_rows", g.groupby("candidate_group").size())
    return t.sort_values("n_rows", ascending=False).reset_index()


def duplicates(df, models):
    """Exact duplicate prompts within and across eval_names (D11)."""
    g = df.assign(candidate_group=df["eval_name"].map(candidate_group))
    evals_per_prompt = g.groupby("prompt")["eval_name"].nunique()
    groups_per_prompt = g.groupby("prompt")["candidate_group"].nunique()
    rows = []
    for e, sub in g.groupby("eval_name"):
        dup_mask = sub["prompt"].duplicated(keep="first")
        dup_sets = sub[sub["prompt"].duplicated(keep=False)].groupby("prompt")
        # A duplicate set "conflicts" if any model's score differs across copies,
        # which would make dropping all-but-one a choice about the label.
        n_conflict = sum(
            any(s[m].astype(float).nunique() > 1 for m in models) for _, s in dup_sets
        )
        rows.append(
            {
                "eval_name": e,
                "candidate_group": candidate_group(e),
                "n_rows": len(sub),
                "n_unique_prompts": sub["prompt"].nunique(),
                "n_dup_rows_within_eval": int(dup_mask.sum()),
                "n_dup_sets_within_eval": dup_sets.ngroups,
                "n_dup_sets_conflicting_scores": int(n_conflict),
                "n_rows_prompt_in_other_eval": int(
                    (sub["prompt"].map(evals_per_prompt) > 1).sum()
                ),
                "n_rows_prompt_in_other_group": int(
                    (sub["prompt"].map(groups_per_prompt) > 1).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def prompt_format(df):
    """How prompts are stored: list-as-string, number of turns, lengths.

    Raw prompts are str(list), so a newline inside a turn appears as the two
    characters backslash-n. Features that count lines or look for
    line-initial "A." only work after parsing; these columns show that.
    `n_unique_turn0` shows whether the first turn is a fixed instruction
    shared by every row of the eval (an easy benchmark fingerprint).
    """
    parsed = df["prompt"].map(parse_prompt)
    ok_mask = parsed.notna()
    work = pd.DataFrame(
        {
            "eval_name": df["eval_name"],
            "raw": df["prompt"],
            "parsed_ok": ok_mask,
            "n_turns": parsed.map(lambda v: len(v) if v is not None else -1),
            "turn0": parsed.map(lambda v: str(v[0]) if v else None),
            "last_turn": parsed.map(lambda v: str(v[-1]) if v else None),
            "total_chars": parsed.map(lambda v: sum(len(str(t)) for t in v) if v is not None else None),
            "newline_in_a_turn": parsed.map(
                lambda v: any("\n" in str(t) for t in v) if v is not None else False
            ),
        }
    )
    rows = []
    for e, sub in work.groupby("eval_name"):
        ok = sub[sub["parsed_ok"]]
        turns = ok["n_turns"].value_counts().sort_index()
        rows.append(
            {
                "eval_name": e,
                "candidate_group": candidate_group(e),
                "n_rows": len(sub),
                "n_parsed_as_list": int(sub["parsed_ok"].sum()),
                "n_parse_failed": int((~sub["parsed_ok"]).sum()),
                "turn_counts": " ".join(f"{k}:{v}" for k, v in turns.items()),
                "n_raw_with_real_newline": int(sub["raw"].str.contains("\n", regex=False).sum()),
                "n_raw_with_literal_backslash_n": int(sub["raw"].str.contains("\\n", regex=False).sum()),
                "n_newline_in_a_turn_after_parse": int(ok["newline_in_a_turn"].sum()),
                "n_unique_turn0": int(ok["turn0"].nunique()),
                "median_chars_turn0": float(ok["turn0"].str.len().median()) if len(ok) else None,
                "median_chars_last_turn": float(ok["last_turn"].str.len().median()) if len(ok) else None,
                "median_chars_all_turns": float(ok["total_chars"].median()) if len(ok) else None,
                "max_chars_all_turns": int(ok["total_chars"].max()) if len(ok) else None,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    if not config.RAW_PICKLE.exists():
        print(
            f"Missing {config.RAW_PICKLE.relative_to(config.ROOT)}.\n"
            "Download it first:  python data/download.py",
            file=sys.stderr,
        )
        return 1

    out = config.DATA_CHECK_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(config.RAW_PICKLE)
    models, roles = column_roles(df)

    schema = schema_table(df, roles)
    evals = eval_counts(df)
    groups = group_counts(df)
    sv = score_values(df, models)
    binary = binary_by_eval(sv, df, models)
    binary_evals = binary.loc[binary["all_models_binary"], "eval_name"]
    summary = model_summary(df, models, binary_evals)
    score_by_group = by_group_wide(df, models, "score")
    cost_by_group = by_group_wide(df, models, "cost")
    dups = duplicates(df, models)
    fmt = prompt_format(df)

    tables = {
        "schema.csv": schema,
        "eval_name_counts.csv": evals,
        "candidate_group_counts.csv": groups,
        "score_values.csv": sv,
        "score_binary_by_eval.csv": binary,
        "model_summary.csv": summary,
        "model_score_by_group.csv": score_by_group,
        "model_cost_by_group.csv": cost_by_group,
        "duplicates.csv": dups,
        "prompt_format.csv": fmt,
    }
    for name, t in tables.items():
        t.to_csv(out / name, index=False)

    info = json.loads(config.DOWNLOAD_INFO.read_text()) if config.DOWNLOAD_INFO.exists() else {}
    overview = {
        "source_file": config.DATASET_FILE,
        "dataset_revision": info.get("revision"),
        "sha256": info.get("sha256"),
        "pandas_version": pd.__version__,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_models": len(models),
        "models": models,
        "n_eval_names": int(df["eval_name"].nunique()),
        "n_candidate_groups": int(groups.shape[0]),
        "sample_id_unique": bool(df["sample_id"].is_unique),
        "n_eval_names_all_binary": int(binary["all_models_binary"].sum()),
        "n_rows_in_all_binary_evals": int(binary.loc[binary["all_models_binary"], "n_rows"].sum()),
        "n_dup_rows_within_eval": int(dups["n_dup_rows_within_eval"].sum()),
        "n_prompts_parse_failed": int(fmt["n_parse_failed"].sum()),
    }
    (out / "overview.json").write_text(json.dumps(overview, indent=2) + "\n")

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 100)
    print(json.dumps(overview, indent=2))
    print(f"\nWrote {len(tables)} CSVs and overview.json to {out.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
