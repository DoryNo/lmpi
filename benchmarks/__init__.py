"""LMPI benchmark — frozen eval set + reproducible detection metrics (Agent 7).

Package layout:

- ``benchmarks.eval_set`` — the frozen manifest (``manifest.json``, committed)
  and the hand-written tricky-benign file (``tricky_benign.jsonl``, committed).
  Attack/clean prompt **texts are never committed** — the manifest stores only
  dataset source + row indices; texts are resolved at run time from HuggingFace
  into the gitignored ``benchmarks/.cache/``.
- ``benchmarks.manifest`` — manifest loading/validation helpers (offline).
- ``benchmarks.hf_sources`` — dataset text resolution (network, cached).
- ``benchmarks.runner`` — pipeline runner + metrics computation (offline core).
- ``benchmarks.run_benchmark`` — CLI entry point.
- ``benchmarks.eval_set.build_manifest`` — one-shot manifest builder (network).
"""

__version__ = "1.0.0"

# Seed used for every deterministic selection in the eval set. Recorded in
# manifest.json as well; re-running build_manifest.py with the same dataset
# revisions reproduces the manifest byte-for-byte apart from frozen_at.
SEED = 20260905
