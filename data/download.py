"""Download the official RouterBench 0-shot pickle into data/raw/.

    python data/download.py

Fetches `routerbench_0shot.pkl` from the Hugging Face dataset
`withmartian/routerbench`, copies it to data/raw/, and records provenance
(SHA-256, retrieval time, dataset revision) in data/raw/DOWNLOAD_INFO.json.

This is one of only three entry points allowed to use the network (PRD 0.10).
The pickle is loaded once at the end as a smoke test. We only ever unpickle
this file from the official repo at a pinned revision, because unpickling
can execute arbitrary code.
"""

import datetime
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

# Run as a script from anywhere: make the repo root importable for `config`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def resolve_revision():
    """Return (revision, was_pinned).

    If config.DATASET_REVISION is unset, resolve the repo's current head to a
    concrete commit sha *before* downloading, then download that exact
    commit. That way the revision we record is guaranteed to be the one the
    file came from, even if the repo is updated mid-download.
    """
    if config.DATASET_REVISION:
        return config.DATASET_REVISION, True
    from huggingface_hub import HfApi

    return HfApi().dataset_info(config.DATASET_REPO).sha, False


def main():
    import huggingface_hub
    from huggingface_hub import hf_hub_download

    revision, was_pinned = resolve_revision()
    print(f"Dataset: {config.DATASET_REPO} / {config.DATASET_FILE}")
    print(f"Revision: {revision} ({'pinned in config' if was_pinned else 'resolved from current head'})")

    cached_path = hf_hub_download(
        repo_id=config.DATASET_REPO,
        filename=config.DATASET_FILE,
        repo_type="dataset",
        revision=revision,
    )

    # The Hugging Face cache stores files read-only and copy2 keeps that mode.
    # We keep data/raw/ read-only on purpose (raw data is never edited), so on
    # a re-run skip the copy if the bytes already match, else replace the file.
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_PICKLE.exists():
        if sha256_of(config.RAW_PICKLE) == sha256_of(cached_path):
            print("data/raw/ already holds this exact file; not copying again.")
        else:
            config.RAW_PICKLE.unlink()
    if not config.RAW_PICKLE.exists():
        shutil.copy2(cached_path, config.RAW_PICKLE)
    digest = sha256_of(config.RAW_PICKLE)
    size = config.RAW_PICKLE.stat().st_size

    # Same revision must mean same bytes; a mismatch means something is wrong
    # with the download or the cache, and we should not silently continue.
    if config.DOWNLOAD_INFO.exists():
        previous = json.loads(config.DOWNLOAD_INFO.read_text())
        if previous.get("revision") == revision and previous.get("sha256") != digest:
            print(
                f"ERROR: SHA-256 {digest} differs from the earlier download "
                f"({previous.get('sha256')}) of the same revision.",
                file=sys.stderr,
            )
            return 1

    info = {
        "repo_id": config.DATASET_REPO,
        "repo_type": "dataset",
        "filename": config.DATASET_FILE,
        "revision": revision,
        "sha256": digest,
        "size_bytes": size,
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "huggingface_hub_version": huggingface_hub.__version__,
        "python_version": platform.python_version(),
    }
    config.DOWNLOAD_INFO.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Saved {config.RAW_PICKLE.relative_to(config.ROOT)} ({size:,} bytes)")
    print(f"SHA-256: {digest}")
    print(f"Provenance: {config.DOWNLOAD_INFO.relative_to(config.ROOT)}")

    # Smoke test: the file must load with the pinned pandas version.
    import pandas as pd

    df = pd.read_pickle(config.RAW_PICKLE)
    print(f"Loaded with pandas {pd.__version__}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    if not was_pinned:
        print(
            "\nNext step: set this in config.py so later downloads get the same file:\n"
            f'    DATASET_REVISION = "{revision}"'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
