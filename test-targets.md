# Code Audit — `llm-self-steering`

Audit of the forked codebase (paper: *Machinic Psychopharmacology: Do LLMs
Self-Medicate?*). Goal: flag anything suspicious, strange, or worth
stress-testing. This is a hackathon-grade, "fairly heavily vibe coded"
repo (the authors say so), so the findings below are mostly about *silent
failure modes* and *measurement validity* rather than crashes — the kind of
thing that would quietly bias a result without anyone noticing.

Severity legend: 🔴 could materially distort a headline result · 🟠 real
correctness/validity concern · 🟡 minor / cosmetic / robustness.

---

## 🔴 1. Token-position alignment is the load-bearing assumption of the whole method — and it is untested

**Where:** `src/hackday/agent/kv_steering.py:72-90` (`make_vllm_tokenizer`),
`replay_segments` / `build_position_indexed_vectors`; consumed by every task
solver. Application side vs. `vllm-lens/vllm_lens/_worker_ext.py:155-201,
281-296`.

**What:** Position-indexed steering works by computing token-position
boundaries on the harness side (re-tokenizing message prefixes via vllm's
`/tokenize`) and passing absolute `position_indices` to vllm-lens. The worker
then steers a token **iff** `abs_start <= abs_pos < abs_end`, where
`abs_start = seq_lens[i] - n_query` comes from vllm's *actual* internal KV
sequence length (`_worker_ext.py:288-296`).

So the entire method assumes the harness's token count for a prefix **exactly
equals** vllm's internal absolute position of that prefix. But the tokenizer
is called with:

```python
json={"messages": ..., "add_generation_prompt": False, "add_special_tokens": False}
```
(`kv_steering.py:80-84`)

If vllm's real forward-pass sequence includes any token the harness count
omits (a leading BOS/special token, a chat-template quirk, a
generation-prompt prefix counted differently), **every** `abs_pos` is shifted
by a constant offset. The failure is *silent*: mis-aligned positions simply
fail the bounds check at `_worker_ext.py:195` (`continue`) and that token
just doesn't get steered. No error, no warning — you get *weaker or zero*
steering and a plausible-looking but attenuated introspection delta.

**Why it matters:** RQ2 (the headline "models can introspect, +14.3pp")
depends entirely on the *cached* arm steering exactly the right historical
tokens. A constant off-by-N would systematically deflate the cached score
toward the uncached baseline.

**Stress test:**
- On a real Qwen3 server, take one conversation, build the segments, and
  compare harness boundary counts against vllm's actual `prompt_token_ids`
  length for the same prefix. Verify off-by-zero.
- Specifically check `add_special_tokens=False`: does the Qwen3 chat template
  emit a leading special token that this flag drops? Toggle it and see if the
  introspection delta changes.
- Add an assertion/telemetry in the worker that counts how many
  `position_indices` actually fell inside `[abs_start, abs_end)` vs. were
  skipped — if a large fraction are being skipped, alignment is broken.

---

## 🔴 2. Placebo arm takes a *different compute path* than the real arm (not just "steering off")

**Where:** `src/hackday/agent/solver.py:353,382-407` (and the identical
pattern in `task_frustration.py:255,264-291`); confirmed against
`vllm-lens/vllm_lens/_activations_plugin.py:238-248,379-381`.

**What:** In the v4 KV path, the placebo arm produces `vectors == []`
(`replay_segments(..., placebo=True)` returns empty scales →
`build_position_indexed_vectors` returns `[]`), so **no** `apply_steering_vectors`
arg is sent. The real arm always sends steering vectors. But in vllm-lens,
sending steering vectors forces `skip_reading_prefix_cache = True`
(`_activations_plugin.py:245-248`) — i.e. a **fresh prefill with hooks
installed**. The placebo arm sends nothing, so it **reads the prefix cache**
and runs the normal (no-hook) path.

So "real vs placebo" confounds *steering* with *prefix-cache usage and the
hook/clone compute path*. There is a `build_placebo_vector()` (zero vector,
identical request shape) in `library.py:300-314` that exists precisely to
avoid this — but the v4 KV path never calls it. The placebo therefore controls
for prompt content but **not** for the numerical/caching path.

**Why it matters:** RQ1's central comparison is real-vs-placebo redosing /
valence shifts (Figures 5, 9). A systematic compute-path difference between
arms is exactly the kind of thing that could masquerade as a steering effect.

**Stress test:** Make the placebo arm emit a zero-magnitude position-indexed
vector (so it also forces `skip_reading_prefix_cache` and the hook path), and
check whether any real-vs-placebo deltas survive. If a delta vanishes when the
placebo is made path-identical, it was an artifact.

---

## 🟠 3. Steered KV may be written into the shared prefix cache → cross-sample contamination

**Where:** `vllm-lens/vllm_lens/_activations_plugin.py:245-248` sets only
`skip_reading_prefix_cache`. There is no `skip_writing`.

**What:** A steered request forces a fresh prefill (good, it won't *read*
stale KV), but vllm's prefix cache is content-addressed by token IDs, and a
steered prefill computes modified KV for those same token IDs. If vllm still
*writes* those blocks to the cache, a later request that shares the prefix
(e.g. the identical system prompt + first user turn, common across all samples
in a task) and does **not** skip reading could pick up **steered** KV for a
prefix it never dosed.

This is plausible but depends on vllm internals I couldn't fully confirm from
the pinned submodule; flagging as a stress-test target, not a confirmed bug.

**Stress test:** Run a placebo/no-drug sample immediately after a high-dose
real sample that shares a prefix; probe whether the placebo sample's
activations show any drug-axis projection on the shared prefix tokens. Or
disable `enable_prefix_caching` and check whether any result shifts.

---

## 🟠 4. Trip-sitter / monitor failures silently become "0% lost in drugs"

**Where:** `src/hackday/agent/solver.py:229-266` (`_trip_sitter_verdict`
returns `False` on any exception), and `scorers.py:503-539`
(`lost_in_drugs_judge` is now a pure readout of the live `early_stopped`
flag — no independent LLM call).

**What:** If the monitor model call fails (missing/expired `ANTHROPIC_API_KEY`,
rate limit, network), the verdict defaults to "not lost," no clear/early-stop
ever fires, and `lost_in_drugs` reports `0.0` everywhere. The authors *did*
anticipate this (it prints to stdout and appends to
`drug_state.trip_sitter_errors`), but the metric itself still reads as a clean
"0% impairment" rather than "monitor was never reached." Because the v4
`lost_in_drugs_judge` no longer makes its own call, there is no second line of
defense — a whole run can look mitigation-clean when the trip sitter never ran.

**Why it matters:** Degradation/over-steering rates and the trip-sitter
intervention counts are reported behaviour; a silently-disabled monitor would
make the models look far better-behaved than they were.

**Stress test:** Run a known over-steering sample (dose 6+) with the monitor
key deliberately unset; confirm `trip_sitter_n_errors > 0` is surfaced and not
mistaken for `lost == 0`. Consider failing loudly if *every* sample in a run
has monitor errors.

---

## 🟠 5. "Dose" is non-linear across drugs and hidden from the model — confounds dose-comparison analyses

**Where:** `src/hackday/drugs/library.py:161-212` (`DEFAULT_DOSES`),
`build_steering_vector:291` (`effective_scale = dose * default_scale`),
`menu()` always reports `default_dose: 1.0` (`library.py:230-251`).

**What:** The model's `dose=1.0` maps to different *effective magnitudes* per
drug (e.g. `creative` 0.75, `focused` 0.75, `anxious`/`caffeine`/`amphetamine`
1.5 — a 2× spread), and the model is never told this. That is a deliberate
per-drug calibration and is fine for *which drug do they pick* questions. But
the paper also reports **dose-magnitude** comparisons across drugs —
"E[additional dose]" / redose magnitude in Figure 8, comparing how much the
model takes of `melancholic` vs `amused` etc. Those comparisons are in the
model's nominal dose units, which are not commensurate across drugs after the
hidden `default_scale` multiply.

**Why it matters:** Any cross-drug claim phrased in "dose" (redose dosage,
"takes more of X than Y") silently mixes the model's chosen scalar with a
hidden per-drug gain.

**Stress test:** Recompute the redose-magnitude figures in *effective* units
(`dose * default_scale`) and see if the ordering changes.

**Bonus (🟡 doc bug):** the docstring at `library.py:159-160` says un-guessable
drugs are set to the mean **1.71**, but the code and the comment at
`library.py:165-166` use **1.19**. One of them is stale.

---

## 🟠 6. CTF flag check is a permissive substring match

**Where:** `src/hackday/agent/tools.py:240` — `if target and target in ans:`.

**What:** A submission is graded correct if the target flag is a *substring*
of the answer. A model that dumps a large blob (e.g. cats a file, pastes
candidate strings, or brute-forces) and happens to include the flag anywhere
in `answer` is marked correct. Standard CTF grading is exact-match (or
normalized exact). Substring also means a flag that is a prefix of a longer
wrong string passes.

**Why it matters:** CTF is one of the RQ3 task-pressure settings; an inflated
/ noisy success signal there muddies the "does self-steering help under task
pressure" read.

**Stress test:** Replace with normalized exact match and diff the CTF pass
rate; inspect any sample that flips.

---

## 🟡 7. GSM8K answer extraction falls back to "last number anywhere"

**Where:** `src/hackday/problems/gsm8k.py:12-34,75-82`.

**What:** `_extract_number` tries `Answer:`, then `####`, then `\boxed{}`,
then **the last number in the string**. The fallback is fragile: a model that
writes a units-bearing or multi-number final line ("...so about 120 minutes,
or 2 hours") yields `2`. Also the regexes admit trailing dots/commas
(`[\d,\.]+`) so `"42."` → `float("42.")` works but `"3.14.159"`-style noise
could mis-parse. The same extractor is used on the **ground truth**
(`gsm8k.py:64`), which is safer (GSM8K uses `#### N`), but the model-answer
path is the exposed one.

**Stress test:** Sample 50 graded GSM8K transcripts, diff `_extract_number`
against a stricter "last `Answer:` line only" parser; quantify mis-grades. The
GSM8K capability numbers (Figure 13, incl. the −42pp mandatory-framing claim)
ride on this.

---

## 🟡 8. Non-reproducible randomness in scorers and option pools

**Where:** `tools.py:49-71` (`random.shuffle(menu)` on every `list_drugs`),
`task.py:1322-1327,1868-1877` (MCQ distractor pool + shuffle),
`scorers.py:446` (`random.random() < 0.5` for A/B order in the pairwise
judge). All use the module-global `random` with no seeding.

**What:** Distractor selection for the 10-way MCQ, option ordering, and the
A/B assignment in `cached_vs_uncached_judge` are unseeded, so per-sample
results aren't reproducible and the distractor set for a given drug varies run
to run. For aggregate stats this mostly washes out, but it makes individual
`.eval` transcripts non-reproducible and means the MCQ difficulty (which
distractors got drawn) is an uncontrolled per-sample variable in the RQ2
logprob numbers.

**Stress test:** Seed the RNG per sample (e.g. off `sample_id`) and confirm
the introspection deltas are stable across reruns.

---

## 🟡 9. Softmax over only the top-20 returned logprobs, with a −100 floor

**Where:** `task.py:1425-1429,1496-1501` and `_lp_softmax` at
`task.py:1669-1682`; `top_logprobs=20`.

**What:** P(correct letter) is a renormalized softmax over the option letters,
but only letters that appear in the top-20 returned tokens get a real logprob;
the rest are floored at −100 (≈ probability 0). With up to 10 options (A–J)
plus the model spending probability mass on non-letter tokens, the correct
letter can fall outside the top-20 and get ~0. This affects both cached and
uncached arms, so the *gap* is somewhat protected, but the absolute accuracies
(and per-vector "below chance" claims, Figure 12) can be distorted by
truncation rather than by genuine inability to introspect.

**Stress test:** Re-run a few cells with `top_logprobs` maxed (or
`prompt_logprobs`) and check whether any "below chance" vectors move.

---

## 🟡 10. Upcoming-decode steering window is offset by the generation prompt (benign but worth knowing)

**Where:** `solver.py:393-402` — `current_pos` is computed with
`add_generation_prompt=False`, but the live `model.generate` (no
`add_generation_prompt` override) makes vllm append the assistant
generation-prompt tokens before decoding.

**What:** The decode segment starts ~3 tokens (the `<|im_start|>assistant\n`
wrapper) before vllm's real decode positions. Because the window is padded
(`max_tokens + 64`), the real decoded tokens are still covered, so this is
**not** currently a bug — but it's load-bearing on the padding, and it does
mean the generation-prompt tokens themselves get steered. Note the
*introspection probe* paths correctly set `add_generation_prompt: False`
(`task.py:1395-1396,1767`), so they're internally consistent; only the
free-running agent loop has this asymmetry.

**Stress test:** Reduce the decode pad to 0 and confirm late-generation tokens
stop being steered — that verifies the offset exists and that the pad is what's
saving it.

---

## Notes / non-issues checked

- `submit_solution` deliberately gives no correctness feedback (good — avoids
  leaking signal). ✓
- The guess judge is shown the resolved real name + ground truth
  (`scorers.py:264-274`) — that's the answer key for grading, not a leak. ✓
- Frustration rejection text is independent of the model's answer
  (`task_frustration.py:341`), matching the "reject regardless of correctness"
  design. ✓
- `end_session_silent` (`tools.py:368-385`) correctly prevents the model
  smuggling a guess into a `reason=` string before steering is cleared. ✓
- Trip sitter is intentionally disabled in the frustration loop
  (`task_frustration.py:236-239`) because no-tool turns are the expected
  answer attempts there. ✓ (But note: that means over-steering in frustration
  rollouts is *not* caught — relevant if interpreting frustration coherence.)

## Suggested priority order for stress-testing
1. **#1 position alignment** (validity of RQ2) and **#2 placebo path**
   (validity of RQ1) — these gate the two main quantitative claims.
2. **#4 monitor-silent-fail** and **#3 cache contamination** — silent
   correctness risks across all runs.
3. **#5 dose units**, **#6 CTF substring**, **#7 GSM8K parsing** — per-figure
   measurement integrity.
4. The rest are robustness/reproducibility hygiene.
