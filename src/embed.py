"""Frozen sentence embeddings for every prompt, cached to data/cache/ (PRD 8.2).

Usage:
    python -m src.embed --limit 50     # quick test on the first 50 prompts
    python -m src.embed                # full run (only once flattening is agreed)

Writes (full run):
    data/cache/emb_minilm.npy   float32, one row per prompt, L2-normalized
    data/cache/emb_ids.npy      the sample_id of each row, same order

Design choices:
- Frozen model only (scope: no fine-tuning). sentence-transformers/all-MiniLM-L6-v2
  with normalize_embeddings=True (D7), so cosine similarity is a plain dot
  product for kNN.
- sentence_transformers is imported INSIDE the functions that need it, never
  at module top, so `python main.py` (which must not use the network and
  must not need this package) never loads it.
- The run is saved in chunks (config.EMBED_CHUNK_SIZE). If it is interrupted,
  rerunning the same command skips finished chunks. A manifest records the
  model and the flattening version, so chunks made with different settings
  are never mixed together.
- This is one of the three entry points allowed to use the network
  (PRD 0.10): it downloads the model from Hugging Face on first use.

Prompt flattening:
Prompts are stored as Python list literals in a string, e.g. "['turn 1', 'turn 2']"
(results/full/data_check/prompt_format.csv). Until Piyush's src/data.py exists
and the team has agreed how to flatten, this file uses a PROVISIONAL rule
(parse the list, join turns with a blank line). The full run is refused while
the rule is provisional, because embeddings of the wrong text would have to be
redone. Reading the pickle directly is the PRD 5.1 workaround for a missing
src/data.py.
"""

import argparse
import ast
import json
import sys
import time

import numpy as np
import pandas as pd

import config

# Set to True only after Avimta and Piyush agree on flattening AND
# flatten_prompt below calls (or matches) src/data.py. Bump FLATTEN_VERSION
# whenever the rule changes, so old cached chunks are not reused.
FLATTENING_AGREED = False
FLATTEN_VERSION = "provisional-v1"

ID_COL = "sample_id"   # from results/full/data_check/schema.csv
PROMPT_COL = "prompt"  # from results/full/data_check/schema.csv


def flatten_prompt(raw):
    """Turn the stored list-string into the text we embed. PROVISIONAL.

    ast.literal_eval parses the list safely (it only accepts literals, never
    runs code) and turns the stored "\\n" escapes into real newlines. Turns
    are joined with a blank line so the model sees them as separate
    paragraphs. If parsing fails, fall back to the raw string rather than
    dropping the prompt (the data check found no parse failures).
    """
    try:
        turns = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return str(raw)
    if isinstance(turns, (list, tuple)):
        return "\n\n".join(str(t) for t in turns)
    return str(turns)


def load_prompts(limit=None):
    """Return (ids, texts) in the pickle's row order.

    Only the id and prompt columns are used; no score, cost or eval_name
    column is touched, so nothing here can leak labels into features.
    """
    if not config.RAW_PICKLE.exists():
        sys.exit(f"{config.RAW_PICKLE} not found. Run `python data/download.py` first.")
    df = pd.read_pickle(config.RAW_PICKLE)[[ID_COL, PROMPT_COL]]
    if limit is not None:
        df = df.head(limit)
    ids = df[ID_COL].astype(str).to_numpy()
    texts = [flatten_prompt(p) for p in df[PROMPT_COL]]
    return ids, texts


def load_model():
    # Imported here, not at the top: see the module docstring.
    from sentence_transformers import SentenceTransformer

    # device="cpu": the brief targets CPU-only machines, and fixing the
    # device keeps runs comparable between Avimta's and Piyush's laptops.
    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def encode(model, texts):
    emb = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=config.EMBEDDING_NORMALIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def check_manifest(chunk_dir, n_prompts):
    """Refuse to resume from chunks made with different settings."""
    manifest = {
        "model": config.EMBEDDING_MODEL,
        "normalize": config.EMBEDDING_NORMALIZE,
        "flatten_version": FLATTEN_VERSION,
        "chunk_size": config.EMBED_CHUNK_SIZE,
        "n_prompts": n_prompts,
    }
    path = chunk_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            sys.exit(
                f"{chunk_dir} holds chunks made with different settings:\n"
                f"  old: {old}\n  new: {manifest}\n"
                "Move or delete that folder yourself, then rerun."
            )
    else:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def run(limit=None):
    if limit is None and not FLATTENING_AGREED:
        sys.exit(
            "Full run refused: prompt flattening is still PROVISIONAL.\n"
            "Agree the rule with Piyush, update flatten_prompt (or call src/data.py),\n"
            "set FLATTENING_AGREED = True, then rerun. For a test, use --limit N."
        )

    # A --limit test writes to its own folder so it can never be mistaken
    # for (or mixed into) the real cache.
    if limit is None:
        chunk_dir = config.EMB_CHUNK_DIR
        out_emb, out_ids = config.EMB_CACHE, config.EMB_IDS
    else:
        test_dir = config.CACHE_DIR / f"test_limit{limit}"
        chunk_dir = test_dir / "chunks"
        out_emb, out_ids = test_dir / "emb_minilm.npy", test_dir / "emb_ids.npy"

    t0 = time.perf_counter()
    ids, texts = load_prompts(limit)
    n = len(ids)
    print(f"Prompts: {n:,} (flattening: {FLATTEN_VERSION})")
    check_manifest(chunk_dir, n)

    model = None  # loaded lazily, so a fully cached rerun needs no download
    size = config.EMBED_CHUNK_SIZE
    starts = range(0, n, size)
    for i, start in enumerate(starts):
        path = chunk_dir / f"chunk_{i:05d}.npy"
        if path.exists():
            print(f"chunk {i + 1}/{len(starts)}: already done, skipping")
            continue
        if model is None:
            model = load_model()
        emb = encode(model, texts[start:start + size])
        # Write to a temp name, then rename: an interrupted write never
        # leaves a half-written chunk that a resume would trust.
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, emb)
        tmp.replace(path)
        print(f"chunk {i + 1}/{len(starts)}: saved {path.name}")

    emb = np.concatenate([np.load(chunk_dir / f"chunk_{i:05d}.npy") for i in range(len(starts))])
    if emb.shape[0] != n:
        sys.exit(f"Row count mismatch: {emb.shape[0]} embeddings for {n} prompts")
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, emb)
    np.save(out_ids, ids)

    elapsed = time.perf_counter() - t0
    print(f"Embeddings: {emb.shape} {emb.dtype}")
    print(f"Time: {elapsed:.1f} s")
    print(f"Wrote {out_emb.relative_to(config.ROOT)} and {out_ids.relative_to(config.ROOT)}")
    return emb, ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="embed only the first N prompts (test run)")
    args = parser.parse_args(argv)
    run(args.limit)


if __name__ == "__main__":
    main()
