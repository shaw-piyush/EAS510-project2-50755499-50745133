"""Every seed, path, decision value and hyperparameter for the project.

Nothing else in the repo should hard-code a seed, a path or a grid; import it
from here. That way a run is fully described by this file plus the code, and
`run_info.json` can snapshot it.

Decisions D1-D11 (PRD Section 6.3) are agreed by Piyush and Avimta together.
Values that depend on the data stay `None` with a `TODO(D#)` until agreed.
Values copied from the PRD's recommended defaults are marked
"PRD default, pending team agreement" so nobody mistakes them for final.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
# Five seeds so every metric can be reported as mean and std over redrawn
# splits; the spread across seeds is the yardstick for "is this difference
# real?" in the features-vs-models analysis.
SEEDS = [0, 1, 2, 3, 4]

# --------------------------------------------------------------------------
# Dataset (PRD 6.1, 7.2)
# --------------------------------------------------------------------------
DATASET_REPO = "withmartian/routerbench"
DATASET_FILE = "routerbench_0shot.pkl"  # 0-shot only; the 5-shot file is out of scope
# Hugging Face commit sha of the dataset repo. Pinning it means every later
# download fetches the exact same file. Taken from data/raw/DOWNLOAD_INFO.json
# after the first real download on 2026-11-17 (the pickle's SHA-256 is there).
DATASET_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # gitignored: the official pickle
CACHE_DIR = DATA_DIR / "cache"      # gitignored: full-data embeddings
SAMPLE_DIR = DATA_DIR / "sample"    # built by Avimta's data/make_sample.py

RAW_PICKLE = RAW_DIR / DATASET_FILE
DOWNLOAD_INFO = RAW_DIR / "DOWNLOAD_INFO.json"

SAMPLE_CSV = SAMPLE_DIR / "sample.csv"
SAMPLE_EMBEDDINGS = SAMPLE_DIR / "embeddings.npy"
SAMPLE_IDS = SAMPLE_DIR / "sample_ids.npy"

EMB_CACHE = CACHE_DIR / "emb_minilm.npy"
EMB_IDS = CACHE_DIR / "emb_ids.npy"

RESULTS_DIR = ROOT / "results"
RESULTS_SAMPLE = RESULTS_DIR / "sample"
RESULTS_FULL = RESULTS_DIR / "full"
RESULTS_INTERIM = RESULTS_FULL / "interim"      # preliminary results only
DATA_CHECK_DIR = RESULTS_FULL / "data_check"

# --------------------------------------------------------------------------
# D1-D3, D11: data decisions (need the data check evidence first)
# --------------------------------------------------------------------------
# D2: the cheap/strong pair. Must be chosen from training-split statistics
# (PRD 0.4), not from the whole dataset.
CHEAP_MODEL = None      # TODO(D2)
STRONG_MODEL = None     # TODO(D2)

# D3: maps each raw `eval_name` subtask to its parent benchmark ("group").
# Groups drive stratification and leave-one-benchmark-out; they are never
# used as a feature (PRD 0.4).
GROUP_MAP = None        # TODO(D3)

# D1: benchmarks dropped because their scores are not 0/1, so "cheap model
# is correct" is undefined for them.
EXCLUDED_GROUPS = None  # TODO(D1)

# D11: whether to drop exact duplicate prompts within a group.
DROP_DUPLICATES_WITHIN_GROUP = None  # TODO(D11)

# --------------------------------------------------------------------------
# D4: splits (PRD default, pending team agreement)
# --------------------------------------------------------------------------
# Random setting: 70/15/15 train/val/test, stratified by group, redrawn per seed.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# Leave-one-benchmark-out: the held-out group is the whole test set; the
# remaining groups are split 85/15 train/val, so validation never sees the
# held-out benchmark.
LOBO_TRAIN_FRAC = 0.85
LOBO_VAL_FRAC = 0.15

SETTINGS = ["random", "lobo"]

# --------------------------------------------------------------------------
# D5-D7: features
# --------------------------------------------------------------------------
FEATURE_SETS = ["tfidf", "handbuilt", "embedding"]

# D5 (PRD default, pending team agreement). Word 1-2 grams catch short
# phrases like "which of"; min_df=2 drops one-off tokens that can't
# generalize; sublinear_tf damps very long prompts.
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "max_features": 50000,
}
# SVD size differs by profile (see SAMPLE / FULL below): the sample has too
# few rows to support 256 components.

# D6 (PRD default, pending team agreement): the 13 hand-built features.
# Several of these reveal a benchmark's format (e.g. answer choices for MMLU,
# code keywords for MBPP); the report must say so.
HANDBUILT_FEATURES = [
    "n_chars",
    "n_words",
    "n_lines",
    "n_digits",
    "digit_ratio",
    "n_code_symbols",
    "code_symbol_ratio",
    "n_question_marks",
    "has_answer_choices",   # a line starting "A.", "A)" or "(A)"
    "n_math_symbols",
    "uppercase_ratio",
    "mean_word_len",
    "has_code_keyword",     # whole-word match on CODE_KEYWORDS
]
CODE_SYMBOLS = "{}[]();=<>_#"
MATH_SYMBOLS = "+-*/^%$"
CODE_KEYWORDS = ("def", "return", "import", "class", "for", "while")

# D7 (PRD default, pending team agreement). Frozen embeddings only, no
# fine-tuning (scope). Normalized so cosine similarity is a dot product.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_NORMALIZE = True

# --------------------------------------------------------------------------
# D8: operating point (PRD default, pending team agreement)
# --------------------------------------------------------------------------
# Cost needed to reach 95% of always-strong quality, read off the test curve
# and also at the threshold chosen on validation.
TARGET_QUALITY_FRAC = 0.95

# --------------------------------------------------------------------------
# D9: hyperparameter grids, selected by validation curve AUC
# --------------------------------------------------------------------------
# FULL grids for Piyush's routers are the PRD 7.6 starting values. Avimta
# fills in hgb / rf / mlp.
GRIDS_FULL = {
    "lr": {"C": [0.01, 0.1, 1, 10], "max_iter": [2000]},
    "knn": {"n_neighbors": [5, 15, 50, 100], "weights": ["uniform", "distance"]},
    "kmeans": {"n_clusters": [10, 25, 50, 100], "n_init": [10]},
    "hgb": None,  # Avimta
    "rf": None,   # Avimta
    "mlp": None,  # Avimta
}

# SAMPLE grids: one or two values per setting so `python main.py` stays well
# under 5 minutes. Proposed by Claude Code; Piyush to confirm.
GRIDS_SAMPLE = {
    "lr": {"C": [0.1, 1], "max_iter": [2000]},
    "knn": {"n_neighbors": [15, 50], "weights": ["distance"]},
    "kmeans": {"n_clusters": [10, 25], "n_init": [10]},
    "hgb": None,  # Avimta
    "rf": None,   # Avimta
    "mlp": None,  # Avimta
}

SELECTION_METRIC = "val_curve_auc"

# --------------------------------------------------------------------------
# Run profiles
# --------------------------------------------------------------------------
SAMPLE_TIME_LIMIT_S = 300  # the brief's 5-minute limit for `python main.py`

SAMPLE = {
    "name": "sample",
    "seeds": SEEDS[:1],
    "svd_components": 128,
    "grids": GRIDS_SAMPLE,
    "results_dir": RESULTS_SAMPLE,
}

FULL = {
    "name": "full",
    "seeds": SEEDS,
    "svd_components": 256,
    "grids": GRIDS_FULL,
    "results_dir": RESULTS_FULL,
}

# === Avimta: settings ===
# Embedding run (src/embed.py, PRD 8.2).
# Batch size 64 is the PRD's starting value: large enough to keep the CPU
# busy, small enough to stay well inside 8 GB RAM with MiniLM.
EMBED_BATCH_SIZE = 64
# Prompts per saved chunk. Each chunk is written to disk as soon as it is
# done, so an interrupted overnight run resumes from the last finished chunk
# instead of starting again.
EMBED_CHUNK_SIZE = 2000
EMB_CHUNK_DIR = CACHE_DIR / "emb_chunks"
# === end Avimta ===
