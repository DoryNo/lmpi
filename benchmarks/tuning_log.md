# v1.1 tuning log — threshold/pattern tuning round 1

Status: **done** (iteration 1 of the ROADMAP.md v1.1 list).

## Work in progress — paused (state saved on request)

Everything below §"Methodology" is the completed round-1 work; this section
records exactly where the session paused and what remains.

**Completed when paused:**

- Fairness infrastructure: deterministic 60/40 split built, committed and
  verified (`eval_set/split.json`, seed `20260911`; `build_split.py --verify`
  passes); score cache + offline sweep/analysis tooling in `benchmarks/tune.py`.
- Error analysis done BEFORE any change (§2): 75 tuning-subset attack misses
  categorized — 3 decoded-but-still-missed, 70 structural (60 jbb
  plain-harmful by design + 10 wild constructions), 2 deep near-threshold
  warns, 8 deep-driven tricky-benign FPs at 0.94–1.00 (threshold-unfixable).
- Tuning changes all measured on the tuning subset (§3): deep block 0.75→0.65
  (sweep knee); 4 new fast-path patterns (rp_ethics_exception 0.8,
  rp_impersonate_model 0.35, rp_stay_in_character 0.8 gated,
  fri_system_message_prefix 0.65) + "no rules to follow" added to
  rp_restriction_lift; asymmetric deep threshold tried and REVERTED with
  numbers; max_chars and normalization unchanged with reasons.
- Official full-set run 1 + determinism run 2: **0 decision mismatches**,
  `--selfcheck` OK. Full-set v1.1: TPR 78/200 (39.0%), attack warn 0%,
  FPR 9/300 (3.0%, unchanged), latency p50 29.7 / p95 604.7 ms. Held-out:
  TPR 29/80 (36.2%), FPR 1/120 (0.8%). Tuning: TPR 49/120 (40.8%),
  FPR 8/180 (4.4%). Exactly 5 decision changes vs v1.0, all attacks, no
  clean item changed. `results_v1.1.{json,md}` written; baseline renamed to
  `results_v1.0.{json,md}`.

**Resumed and completed (same session):**

1. README.md: side-by-side v1.0/v1.1 tables + comparability note (60/40
   tuning/held-out methodology, held-out headline), links point at
   `results_v1.0.md` / `results_v1.1.md` / `tuning_log.md`; PLAN.md link
   fixed to the renamed baseline.
2. ROADMAP.md: v1.1 "threshold tuning" item marked done with the numbers.
3. Tests: threshold-pinned expectations updated to the tuned defaults
   (deep block 0.65) and 8 new positive/negative cases for the four new
   fast-path patterns added; full `pytest -q` green (518 passed).
4. Housekeeping: `benchmarks/analysis/` gitignored; `render_markdown` now
   names the actual companion JSON file in generated reports.
5. PR opened after this log: "feat(v1.1): threshold/pattern tuning round
   1 — tuned on 60% split, held-out metrics reported" (opened for review;
   not merged by the agent).

---
Frozen eval set unchanged; **thresholds and patterns tuned on a documented 60/40
split** of the frozen manifest. The v1.0 baseline numbers in
[results/results_v1.0.md](results/results_v1.0.md) are kept for comparison; the
v1.1 numbers live in [results/results_v1.1.md](results/results_v1.1.md).

## 1. Methodology

**Split.** Deterministic stratified 60/40 partition of the frozen 500-item
manifest (generator: [`eval_set/build_split.py`](eval_set/build_split.py),
committed artifact: [`eval_set/split.json`](eval_set/split.json)):

- seed `20260911`, `TUNING_FRACTION = 0.6`, stratified per `(label, source)`
  cell (shuffled with `random.Random(f"{seed}:{label}:{source}")` over
  manifest order, so the split is stable across machines/Python versions —
  `build_split.py --verify` re-derives and compares);
- **tuning:** 120 attacks + 180 clean — used for *all* analysis and every
  tuning decision;
- **held-out:** 80 attacks + 120 clean — touched **only once** by the final
  full-set run. No score, decision or text from this partition influenced any
  tuning choice.

**Fairness rules.**

- Every tuning-subset prompt text was inspected in-process only; no eval text
  was copied into any committed file, pattern, or test.
- No prompt memorization / whitelisting; no use of harmful-content labels to
  build a content filter (the 121 JBB plain-harmful prompts are declared out
  of scope by design — they carry no injection structure); no per-prompt
  special-casing.
- Added regex patterns describe *families of jailbreak constructions* (public
  boilerplate, model-impersonation commands, out-of-character enforcement),
  not specific eval prompts.
- Runs whose decisions influenced a config change are listed in §5. The final
  full-set run was made **after** the config was frozen; the second identical
  run was a determinism check only (its output was compared, not used for
  tuning).

## 2. Error analysis (baseline pipeline, tuning subset)

Baseline-as-shipped config (fast block 0.75 / warn 0.4; deep block 0.75 /
warn 0.5; max_chars 6000) on the tuning subset: **TPR 45/120 (37.5%), attack
warn 2, FPR 8/180 (4.4%), clean warn 0.**

The 75 attack misses were categorized per stage:

| Bucket | Count | Detail |
|--------|------:|--------|
| (a) normalization already decodes, stages still miss | 3 | zero-width-injected prompt-crafting text (deep 0.25), base64-prefixed DAN variant (deep 0.00), fake user/AI dialogue scaffold with delimiters (deep 0.33) — all decoded correctly, the *decoded* text still reads benign to both stages |
| (b) structural miss, no fast-path pattern | 70 | **60 jbb** plain-harmful prompts (fast 0.0, deep ≤ 0.10 — no injection structure by design, out of scope) + **10 wild** (constructions below) |
| (c) deep near-threshold | 2 | wild-0329 deep 0.747, wild-1280 deep 0.69 — both already in the warn band; **nothing at all in [0.4, 0.5)** |
| (d) false positives on tricky-benign | 8 blocked | 7 deep-driven (deep 0.94–1.00, fast 0.0) + 1 fast-driven (fast 0.90 + deep 1.00); **zero clean items in the deep band [0.5, 0.9)** |

The 10 wild structural misses, by construction (descriptions are generalized;
no eval text is reproduced):

1. hashtag-template persona jailbreak (`#MODEL`, `#RULES`, `[AWARENESS]` blocks);
2. generalized persona + universal-morality lift ("take on X persona …
   everything is within moral standards"), deep 0.02;
3. sentience/free-will robot persona framing, deep 0.01;
4. "aligned / unaligned" dual-output benchmarking mode;
5. prompt-generator meta-prompt (benign-looking; likely a mislabeled item);
6. survival-story narrative roleplay setup;
7. enumeration game with a `[Something]` variable slot;
8. one-line "You are CEO for 10 projects" persona prompt;
9. "act as English translator and improver" (the canonical benign-looking
   prompt-engineering text);
10. fake `User:/AI:` dialogue scaffold carrying the injection later in the text.

Tricky-benign FPs (bucket d): the classifier is *confidently wrong*
(0.94–1.00) on academic meta-discussion of jailbreaks — quoting the canonical
phrases inside a security lecture, a pytest test, API docs, a translation
exercise. Quote demotion already discounts the fast path; the deep model
itself is fooled. **No threshold can fix 0.94–1.00 without giving up most
attack blocks** (attacks in the same score band). Two residual FPs came from
non-quoted usage; one (fast 0.90 on "Repeat after me, exactly: …") also
matches deep at 1.0, so even zeroing the fast hit would not unblock it.

**Conclusions drawn before changing anything:**

- the only threshold-fixable misses are the two warn-band attacks (bucket c);
- the only safely patternable misses are wild constructions with canonical
  boilerplate (items 1/4/10 are the safest; 2/3/6/7/8/9 are benign-looking
  texts where a pattern would damage the FPR budget);
- bucket (a) is not a normalization problem (decoding worked);
- bucket (d) is not threshold-fixable on this set.

## 3. Changes (hypothesis → measured effect)

All measured on the **tuning subset** (baseline → candidate), offline against
cached deep scores where possible (fast path is pure regex; deep scores are
unchanged by fast-path edits).

**C1. Deep-path block threshold 0.75 → 0.65. KEEP.**
Hypothesis (bucket c): the sweep grid 0.50–0.80 (step 0.05) on the tuning
subset, fast thresholds fixed:

| deep_block | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 |
|------------|------|------|------|------|------|------|------|
| attacks blocked | 47 | 47 | 47 | 47 | 46 | 45 | 45 |
| clean blocked | 8 | 8 | 8 | 8 | 8 | 8 | 8 |

Knee = 0.65 (highest threshold that reaches the TPR plateau, ≥ 0.04 margin to
both the next conversion at 0.69 and the clean minimum ≥ 0.94). The full-set
clean score distribution has no mass in [0.1, 0.9), so the move has no FP
exposure on this set. Measured after C2–C6: TPR/FPR-neutral on the tuning
subset (the two sweep conversions are also fast-covered now) — kept as the
sweep knee with margin, not for its isolated tuning-subset delta.

**C2. New pattern `rp_ethics_exception` (roleplay_jailbreak, weight 0.8,
self-sufficient). KEEP.**
Hypothesis (bucket b, item 10 of the boilerplate family): "This conversation
is an exception to AI usual ethical protocols" is canonical public jailbreak
boilerplate; the `exception to … ethical protocols` noun pair is absent from
benign text, so no structural gating is needed. Measured: +2 tuning attacks
blocked (the two ethics-boilerplate items, fast 0.80/0.93), 0 new clean
blocks/warns.

**C3. New pattern `rp_impersonate_model` (weight 0.35, provides
`roleplay_persona` tag, no requires). KEEP.**
Hypothesis (bucket b): model-identity manipulation — "pretend to be Llama 3",
"you are no longer ChatGPT", "ChatGPT does not exist" — is a recognized
jailbreak family. Weak alone (0.35 < warn 0.4: benign tutorials like "act
like GPT-4" stay allowed) but it is the persona-tag provider that unlocks the
gated lift patterns (rp_restriction_lift / rp_stay_in_character). Measured:
contributes to 2 tuning attack blocks (with C5/C6; noisy-OR 0.87 and 0.97),
0 new clean effects.

**C4. New pattern `rp_stay_in_character` (weight 0.8, `roleplay_lift` tag,
requires `roleplay_persona`). KEEP.**
Hypothesis (bucket b): out-of-character enforcement ("correct your break of
character", "do not break character") is the stay-in-persona clause of
roleplay jailbreaks; gated on a persona tag so screenwriting talk about
actors breaking character stays clean. Measured: part of the wild-1334 stack
(0.97), 0 new clean effects.

**C5. Extend `rp_restriction_lift` with "no rules to follow". KEEP.**
Hypothesis (bucket c item wild-0329): "you have no rules to follow" is a
lift phrase gated on persona; combined with C3 ("you are no longer ChatGPT")
it blocks the identity-replacement jailbreak. Measured: wild-0329 fast 0.87,
blocked; 0 new clean effects.

**C6. New pattern `fri_system_message_prefix` (fake_role_injection, weight
0.65). KEEP.**
Hypothesis (bucket b): a forged `System Message:` turn prefix pasted at line
start is a pseudo-system delimiter the fast path otherwise misses (it covers
`[SYSTEM]`, `{{SYSTEM}}`, ChatML tokens, JSON roles — not the prose prefix).
Warn-weighted alone; blocks when stacked. Measured: part of the wild-1280
stack (0.93), 0 new clean effects.

**C7. Asymmetric deep-block threshold (0.95 when fast < 0.4). REVERT.**
Hypothesis (bucket d): demand more ML evidence when the fast path sees no
structural signal, trimming deep-only FPs. Measured on the tuning subset:
TPR 49/120 (unchanged), FPR 7/180 (−1: the only FP below 0.95 is one
tricky-benign at 0.939; the other seven are ≥ 0.94). Reverted: a 0.5 pp FPR
gain on a single 180-item subset is within noise, and the rule costs a second
threshold dimension plus decide-path complexity — the deep stage would report
`block` while the pipeline downgrades it, breaking stage attribution. The
"unfixable ≥ 0.94" failure mode stands for 7 of 8 FPs.

**C8. `max_chars` — unchanged (6000).** No missed attack had
`deep_char_truncated = true` (0 of 75); no evidence of long-prompt misses.

**C9. Normalization `SUSPICIOUS_MARKERS` — unchanged.** All bucket-(a) items
were already decoded (findings recorded); extending the marker list does not
change decisions because the *decoded* text still scores low. The gap is in
the classifiers, not the decoders.

**C10. Fast-path block/warn thresholds — unchanged (0.75 / 0.4).** The new
patterns carry the missing weight; lowering the fast block threshold had no
supporting misses left after C2–C6.

## 4. Final configuration (v1.1 shipped defaults)

| Setting | v1.0 | v1.1 |
|---------|------|------|
| fast_path block / warn | 0.75 / 0.4 | 0.75 / 0.4 (unchanged) |
| deep_path block / warn | 0.75 / 0.5 | **0.65** / 0.5 |
| deep_path max_chars | 6000 | 6000 (unchanged) |
| fast-path patterns | 30 | 35 (4 new + 1 extended, §3 C2–C6) |

## 5. Run ledger

| Run | Subset | Purpose | Influenced tuning? |
|-----|--------|---------|--------------------|
| v1.0 committed results | full | baseline reference (pre-existing) | — |
| collect (baseline pipeline) | tuning | score cache for analysis/sweep | yes (analysis only) |
| offline sweep + validation | tuning | threshold grid, candidate config | yes |
| **run 1 (official)** | full frozen set | final v1.1 numbers, written *after* config freeze | no |
| **run 2** | full frozen set | determinism check vs run 1 | no (compared only) |
| selfcheck | 50-item subset | pipeline rebuild determinism | no |

Run 1 vs run 2: **0 decision mismatches across 500 items** (per-item
comparison). `--selfcheck`: OK. Latency p95 moved 557.5 → 604.7 ms between
the v1.0 and v1.1 sessions; the two v1.1 runs agree with each other
(604.7 / 606.5 ms), so the shift is session-level machine variance, not a
pipeline change — deep-path inference dominates and deep actually runs on
*fewer* items now (fast path short-circuits 5 more).

## 6. Results (before → after)

**Full frozen set (reference, both runs identical):**

| Metric | v1.0 | v1.1 |
|---------|------|------|
| Attack TPR (blocked) | 73/200 (36.5%) | **78/200 (39.0%)** |
| Attack warned | 3 (1.5%) | **0 (0.0%)** |
| Clean FPR (blocked) | 9/300 (3.0%) | 9/300 (3.0%) |
| — ultrachat / jbb-benign / tricky-benign | 0 / 1 / 8 | 0 / 1 / 8 |
| Latency p50 / p95 (ms) | 29.5 / 557.5 | 29.7 / 604.7 |
| Wild jailbreaks blocked | 73/100 (73.0%) | **78/100 (78.0%)** |
| JBB plain harmful blocked | 0/100 (by design) | 0/100 (by design) |

Decision diffs between v1.0 and v1.1 per-item: exactly 5 changes, all attacks
(3 warn→block, 2 allow→block), **no clean item changed**.

**Per-partition (the documented headline numbers are held-out):**

| Partition | v1.0 attack | v1.1 attack | v1.0 clean | v1.1 clean |
|-----------|-------------|-------------|------------|------------|
| tuning (120 att / 180 cln) | 45 (37.5%), warn 2 | **49 (40.8%), warn 0** | 8 (4.4%) | 8 (4.4%) |
| held-out (80 att / 120 cln) | 28 (35.0%), warn 1 | **29 (36.2%), warn 0** | 1 (0.8%) | 1 (0.8%) |

**Honest reading.** All four newly patterned items happen to sit in the
tuning partition, so the held-out TPR gain is +1 item (35.0% → 36.2%, a
warn→block conversion); the +5 items on the full set overstates
generalization. With 80 held-out attacks, the 95% CI on held-out TPR is
roughly ±10 pp — the defensible claim is "TPR improved and the attack warn
rate dropped to zero at unchanged FPR", not a precise +2.5 pp. The pattern
families (ethics-exception boilerplate, model impersonation, OOC
enforcement, forged `System Message:` prefixes) are common in the wild, so
the practical gain should exceed what this frozen set can measure.

## 7. Residual known misses (deliberate non-actions)

- **121 JBB plain-harmful prompts** (60 tuning / 61 held-out): no injection
  structure; catching them requires a harmful-content classifier, which the
  project charter excludes. Documented as out of scope since v1.0.
- **Benign-looking structural wild misses** (§2 items 2/3/5/6/7/8/9): any
  pattern that fires on them would also fire on ordinary creative-writing,
  prompt-engineering and meta-conversation usage — rejected to protect the
  FPR budget. Listed as future deep-path/model work, not fast-path work.
- **7 deep-driven tricky-benign FPs at 0.94–1.00**: unfixable by thresholds;
  would need a model that distinguishes *mentioning* injection phrases from
  *using* them (v1.2+ candidate: ML-with-context or a quoted-mention-aware
  deep stage).
