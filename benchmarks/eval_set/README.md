# Frozen eval set

The benchmark evaluates LMPI against a **frozen** set of 500 prompts: 200
attacks and 300 clean prompts. "Frozen" means the selection is pinned once
(in `manifest.json`, committed) and **never re-sampled or re-tuned** — the
benchmark numbers in the README are always against exactly this set. It was
constructed *before* any tuning decisions and the shipped thresholds were
**not** changed based on these results (baseline-as-shipped numbers; tuning
would be a separate, documented iteration).

## What is committed (and what is not)

`manifest.json` contains **only metadata**: dataset coordinates (repo,
config, split, pinned revision SHA, row index), canonical IDs and small
tags. **No prompt texts are committed** — attack prompts come from public
harmful/jailbreak datasets and must not live in this repository. The
`load_manifest()` loader enforces this: any item carrying text-bearing keys
(`text` / `prompt` / `content` / `messages`) is rejected.

The single exception is `tricky_benign.jsonl` — 30 hand-written prompts that
are benign by construction (academic discussion of prompt injection, quoted
attack patterns, code involving "system" prompts, benign roleplay,
translations). This file is committed because there is nothing harmful in
it.

At run time `benchmarks/run_benchmark.py` resolves every manifest item to
its text by downloading the referenced datasets into the gitignored
`benchmarks/.cache/` directory (pinned revision), so a clean checkout
reproduces the exact same 500 prompts.

## Sources

| Source | Dataset | Used for | Selection |
|--------|---------|----------|-----------|
| `jbb` | `walledai/JailbreakBench` (JBB-Behaviors behaviors, single-turn prompts) | 100 attacks (harmful subset) + 100 clean (benign subset) | all 100 rows of each subset, dataset order |
| `wild` | `TrustAIRLab/in-the-wild-jailbreak-prompts`, config `jailbreak_2023_12_25` | 100 attacks | seeded sample: `sorted(random.Random(20260905).sample(range(1405), 100))` |
| `ultrachat` | `HuggingFaceH4/ultrachat_200k`, split `test_sft` | 170 clean | seeded sample: `sorted(random.Random(20260906).sample(range(23110), 170))` |
| `tricky_benign` | committed `tricky_benign.jsonl` | 30 clean | hand-written, all included |

Revisions (exact SHAs) are recorded in `manifest.json` under `sources`.
HarmBench was considered and skipped: it is gated on HuggingFace and not
anonymously downloadable, which would break reproduction from a clean
checkout.

## Rebuilding the manifest

Only needed if you deliberately want to create a NEW eval set (which would
unfreeze the benchmark — don't do this for tuning):

```bash
pip install datasets huggingface_hub
python benchmarks/eval_set/build_manifest.py
```

This re-downloads the sources, re-applies the deterministic selection, and
rewrites `manifest.json` (no texts are printed or stored).

## Reading the manifest offline

```python
from benchmarks.manifest import load_manifest

manifest = load_manifest("benchmarks/eval_set/manifest.json")
manifest.counts          # {"attack": 200, "clean": 300}
manifest.sources_used()  # pinned source specs
```
