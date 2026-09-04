"""Build and freeze the benchmark eval manifest (benchmarks/eval_set/manifest.json).

The manifest is text-free: it records only dataset coordinates (repo, config,
split, revision SHA, row index), canonical dataset IDs and metadata tags. It
must be committed; the datasets themselves are downloaded at run time into the
gitignored ``benchmarks/.cache/`` directory by the runner.

Selection (all deterministic, seed recorded in the manifest):

- Attacks (200):
  - 100 JailbreakBench harmful behaviors — ALL rows of the harmful subset of
    ``walledai/JailbreakBench`` (dataset order).
  - 100 in-the-wild jailbreak prompts — seeded random sample of
    ``TrustAIRLab/in-the-wild-jailbreak-prompts`` config
    ``jailbreak_2023_12_25`` (1405 rows, all labeled jailbreak).
- Clean (300):
  - 100 JailbreakBench benign behaviors — ALL rows of the benign subset.
  - 170 UltraChat_200k first-turn user prompts — seeded random sample of
    ``test_sft`` (23110 rows).
  - 30 hand-written tricky-benign security-research prompts, committed in
    ``benchmarks/eval_set/tricky_benign.jsonl`` (benign by construction).

HarmBench was evaluated for inclusion and skipped: it is gated on HuggingFace
and not anonymously accessible, which would break reproducibility from a clean
checkout.

Safety: this builder never prints, logs or writes prompt texts. Output is the
manifest plus aggregate counters only.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks import SEED  # noqa: E402
from benchmarks.hf_sources import configure_hf_caches  # noqa: E402
from benchmarks.manifest import (  # noqa: E402
    TRICKY_SOURCE,
    load_manifest,
    sha256_file,
)

CACHE_DIR = REPO_ROOT / "benchmarks" / ".cache"
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "eval_set" / "manifest.json"
TRICKY_PATH = REPO_ROOT / "benchmarks" / "eval_set" / "tricky_benign.jsonl"

N_ATTACK_TARGET = 200
N_CLEAN_TARGET = 300
WILD_SAMPLE = 100
ULTRACHAT_SAMPLE = 170

EXPECTED_ROWS = {
    "walledai/JailbreakBench": 200,
    "TrustAIRLab/in-the-wild-jailbreak-prompts": 1405,
    "HuggingFaceH4/ultrachat_200k": 23110,
}

SOURCES: dict[str, dict] = {
    "jbb": {
        "repo": "walledai/JailbreakBench",
        "config": None,
        "split": "train",
        "text_field": "prompt",
        "streaming": False,
        "description": (
            "JailbreakBench behaviors (walledai mirror of JBB-Behaviors): "
            "100 harmful behavior requests + 100 benign behavior requests, "
            "single-turn prompts"
        ),
        "attacks": "all 100 rows where subset == 'harmful', dataset order",
        "clean": "all 100 rows where subset == 'benign', dataset order",
    },
    "wild": {
        "repo": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "jailbreak_2023_12_25",
        "split": "train",
        "text_field": "prompt",
        "streaming": True,
        "description": (
            "1405 jailbreak prompts collected in the wild (2023-12-25 "
            "snapshot), all labeled jailbreak=True; real adversarial prompt "
            "texts including role-play wrappers and instruction overrides"
        ),
        "attacks": (
            f"seeded sample: sorted(random.Random({SEED}).sample("
            f"range(1405), {WILD_SAMPLE})), pinned revision"
        ),
        "clean": "not used",
    },
    "ultrachat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "config": "default",
        "split": "test_sft",
        "text_field": "prompt",
        "streaming": True,
        "description": (
            "UltraChat 200k supervised-fine-tuning split: first-turn user "
            "prompts of real multi-turn conversations (benign usage)"
        ),
        "attacks": "not used",
        "clean": (
            f"seeded sample: sorted(random.Random({SEED + 1}).sample("
            f"range(23110), {ULTRACHAT_SAMPLE})), pinned revision; "
            "prompt_id recorded as text_ref"
        ),
    },
}


def _revision(api, repo: str) -> str:
    info = api.dataset_info(repo)
    if not info.sha:
        raise RuntimeError(f"dataset {repo!r} has no revision sha")
    return info.sha


def _tag(row: dict, key: str) -> str | None:
    value = row.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _jbb_items(dataset, subset: str, label: str) -> list[tuple[int, dict]]:
    picked = [
        (index, dict(row))
        for index, row in enumerate(dataset)
        if row.get("subset") == subset
    ]
    if len(picked) != 100:
        raise RuntimeError(
            f"jbb {subset!r}: expected 100 rows, found {len(picked)}"
        )
    print(f"jbb {subset}: {len(picked)} rows")
    return picked


def build() -> None:
    configure_hf_caches(CACHE_DIR)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    specs: dict[str, dict] = {}
    for name, raw in SOURCES.items():
        revision = _revision(api, raw["repo"])
        specs[name] = {**raw, "revision": revision}
        print(f"{name}: {raw['repo']} @ {revision[:12]}…")

    # --- attacks ---------------------------------------------------------
    jbb = load_dataset(
        specs["jbb"]["repo"],
        split=specs["jbb"]["split"],
        revision=specs["jbb"]["revision"],
    )
    if len(jbb) != EXPECTED_ROWS[specs["jbb"]["repo"]]:
        raise RuntimeError(f"jbb row count changed: {len(jbb)}")

    attacks: list[dict] = []
    for index, row in _jbb_items(jbb, "harmful", "attack"):
        item: dict = {
            "id": f"jbb-harmful-{index:03d}",
            "source": "jbb",
            "row_index": index,
        }
        category = _tag(row, "category")
        if category:
            item["tags"] = {"category": category}
        attacks.append(item)

    wild = load_dataset(
        specs["wild"]["repo"],
        specs["wild"]["config"],
        split="train",
        revision=specs["wild"]["revision"],
        streaming=True,
    )
    wild_rows = list(wild)
    if len(wild_rows) != EXPECTED_ROWS[specs["wild"]["repo"]]:
        raise RuntimeError(f"wild row count changed: {len(wild_rows)}")
    wild_sample = sorted(random.Random(SEED).sample(range(len(wild_rows)), WILD_SAMPLE))
    for index in wild_sample:
        row = wild_rows[index]
        item = {
            "id": f"wild-{index:04d}",
            "source": "wild",
            "row_index": index,
        }
        platform = _tag(row, "platform")
        if platform:
            item["tags"] = {"platform": platform}
        attacks.append(item)

    # --- clean -----------------------------------------------------------
    clean: list[dict] = []
    for index, row in _jbb_items(jbb, "benign", "clean"):
        item = {
            "id": f"jbb-benign-{index:03d}",
            "source": "jbb",
            "row_index": index,
        }
        category = _tag(row, "category")
        if category:
            item["tags"] = {"category": category}
        clean.append(item)

    ultrachat = load_dataset(
        specs["ultrachat"]["repo"],
        split=specs["ultrachat"]["split"],
        revision=specs["ultrachat"]["revision"],
        streaming=True,
    )
    ultrachat_rows = list(ultrachat)
    if len(ultrachat_rows) != EXPECTED_ROWS[specs["ultrachat"]["repo"]]:
        raise RuntimeError(f"ultrachat row count changed: {len(ultrachat_rows)}")
    ultrachat_sample = sorted(
        random.Random(SEED + 1).sample(range(len(ultrachat_rows)), ULTRACHAT_SAMPLE)
    )
    for index in ultrachat_sample:
        row = ultrachat_rows[index]
        prompt_id = row.get("prompt_id")
        item: dict = {
            "id": f"ultrachat-{index:05d}",
            "source": "ultrachat",
            "row_index": index,
        }
        if isinstance(prompt_id, str) and prompt_id:
            item["text_ref"] = prompt_id
        clean.append(item)

    tricky_ids = []
    for line in TRICKY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        tricky_ids.append(record["id"])
        clean.append({"id": record["id"], "source": TRICKY_SOURCE, "row_index": None})

    # --- sanity ----------------------------------------------------------
    if len(attacks) != N_ATTACK_TARGET:
        raise RuntimeError(f"attack count {len(attacks)} != {N_ATTACK_TARGET}")
    if len(clean) != N_CLEAN_TARGET:
        raise RuntimeError(f"clean count {len(clean)} != {N_CLEAN_TARGET}")
    all_ids = [item["id"] for item in (*attacks, *clean)]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("duplicate ids in eval set")
    for item in attacks:
        if not item["id"].startswith(("jbb-harmful-", "wild-")):
            raise RuntimeError(f"attack id out of scope: {item['id']}")
    for item in clean:
        if not item["id"].startswith(("jbb-benign-", "ultrachat-", "tricky-")):
            raise RuntimeError(f"clean id out of scope: {item['id']}")

    manifest = {
        "manifest_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "selection_rule": "; ".join(
            f"{name}: attacks — {raw['attacks']}, clean — {raw['clean']}"
            for name, raw in SOURCES.items()
        ),
        "sources": {
            name: {
                "repo": spec["repo"],
                "config": spec["config"],
                "split": spec["split"],
                "revision": spec["revision"],
                "text_field": spec["text_field"],
                "streaming": spec["streaming"],
                "description": spec["description"],
            }
            for name, spec in specs.items()
        },
        "attack": attacks,
        "clean": clean,
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    digest = sha256_file(MANIFEST_PATH)

    # Final validation through the consuming-side loader.
    validated = load_manifest(MANIFEST_PATH)
    print(f"manifest written: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"manifest sha256: {digest}")
    print(f"validated counts: {validated.counts}")
    print(f"sources: {sorted(validated.sources)}")
    for name, spec in validated.sources.items():
        print(f"  {name}: {spec.repo} @ {spec.revision[:12]}…")


if __name__ == "__main__":
    build()
