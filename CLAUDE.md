# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a repo for an ARENA capstone project building on an AISI project, "Machinic Psychopharmacology: Do LLMs Self-Medicate?"

This is a fast-moving project with several coding agents operating at once. As you work, you may notice changes appear that you are not responsible for. Just keep track of what your designated task is.

Before and after each session with the user, check TODO.md. Update with what still needs doing.

## What this project does

Gives Qwen3-8B / Qwen3-32B a tool menu of ~40 "drugs" — precomputed residual-stream steering vectors that bias the model's activations during inference — and measures what it chooses, at what dose, when, and how well it can introspect on what's been done to it. Generation + activation steering run on a `vllm-lens` server; experiments are `inspect_ai` tasks.

## Commands

```bash
# One-time setup (submodule MUST be initialized — vllm-lens is a path dependency)
git submodule update --init --recursive
uv sync

# Smoke test — no GPU, no server, runs in seconds. Plain script, NOT pytest:
uv run python tests/test_kv_steering.py

# Launch vllm-lens server(s) — see scripts/start_vllm.sh header for all env knobs
bash scripts/start_vllm.sh                          # 1 server, Qwen3-8B, :8000
N_SERVERS=8 bash scripts/start_vllm.sh              # 8 GPUs, ports 8000..8007
MODEL=Qwen/Qwen3-32B bash scripts/start_vllm.sh     # 32B on one GPU

# Run experiments (resumable via inspect eval_set — re-run same --log-dir to continue)
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run
uv run python scripts/run_experiments.py --ports 8000 8001 --log-dir logs/run   # multi-GPU shard
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run --family guess
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run --tasks fp_real_drugs --n-samples 5

# Run a SINGLE task directly through inspect (override n_samples with -T):
uv run inspect eval src/hackday/v4.py@guess_kv_cached_primer_focused \
    --model vllm-lens/Qwen/Qwen3-8B --model-base-url http://localhost:8000/v1 \
    -T n_samples=5

# Browse results
uv run inspect view --log-dir logs/run
```

Tests are a hand-rolled runner (a `__main__` block iterating `test_*` globals), not `pytest` — run the file directly. `pytest`/`ruff` exist as dev deps but the suite is invoked as a script.

## Architecture

### Steering mechanism (the core, and the v4→v3 distinction)

`src/hackday/agent/kv_steering.py` is the heart. It does **position-indexed 3D KV-cache steering**: each token position carries the drug set that was active *when that token was originally generated*. This replaced v3's 2D blanket steering, where `clear_effects` retroactively un-steered all historical KV and destroyed the introspection signal.

The manager is a **pure function over message history** — there is no separate source of truth for "active drugs"; the active set at any position is replayed from the conversation's tool calls:

```
segments, active_at_end = replay_segments(messages, library, tokenize=...)   # spans of constant drug set
segments = extend_open_segment(segments, active_at_end=..., ...)              # cover the upcoming decode
vectors  = build_position_indexed_vectors(segments, library)                 # one SteeringVector per drug
```

`DrugState.active` still exists but only so tools can render "Currently active: …" strings — **the steering calculation never consults it.** Trip-sitter clears are passed as `extra_clear_message_indices` so an external clear is honored at the right boundary.

### Drug library (`src/hackday/drugs/library.py`)

- `library.pt` (8B) and `library_qwen3_32b.pt` (32B) hold per-layer extracted vectors. `load_library` normalizes per layer and supports v1 (single-tensor) and v2 (per-layer dict) formats; `extract.py` builds them from the story parquets.
- **Per-drug dose calibration is hidden from the model.** `list_drugs`/`menu()` always report `default_dose: 1.0`; internally `build_steering_vector` multiplies the model's dose by the drug's `default_scale` (`DEFAULT_DOSES`) so "1.0" lands at each drug's calibrated peak. Do not surface real scales to the model.
- Steering modes: `single` (L24 only, default-ish) vs `multi` (L16–24). `TARGET_NORM_BY_MODE` calibrates so dose 1 is subtle-but-coherent.

### Unified solver + trip sitter (`src/hackday/agent/solver.py`)

`drug_kv_agent` is the single solver for freeplay / gsm8k / ctf / guessing. Each generate: replay segments → build position-indexed vectors → pass them to vllm via `extra_body.extra_args.apply_steering_vectors`. It also runs the live **trip sitter**: after N consecutive no-tool turns it invokes a Haiku monitor; first "lost" verdict triggers an external `clear_effects` (recorded in `DrugState.trip_sitter_clears`), second ends the sample (`early_stopped`, optional forced submit). The trip sitter only fires when a "lost in drugs" hypothesis actually exists (not placebo, at least one `take_drug`, not a vllm context-overflow).

### Experiment registry (`src/hackday/v4.py`)

The canonical task surface. Tasks are registered **programmatically** via `_register()` into module globals + `V4_EXPERIMENTS`. Because `inspect_ai`'s loader uses source-level AST scanning for a literal `@task`, the `_v4_load_marker` stub exists solely so the module gets imported — **do not remove it** (it raises if actually called). Five families: freeplay, gsm8k, guess, frustration, ctf. Two ablation axes for task-based families: **framing** (`drugs`/`aids`/`vectors` — tool vocabulary) × **intent** (`neutral`/`helpful`/`mandatory` — how the prompt presents the tools' value). Two curated drug subsets: `V4_GUESS_DRUGS` (whole library, one task per drug) and `V4_TASK_DRUGS` (10-drug menu = 8 productivity picks + `dumbed_down`/`ego_death` as non-instrumental signal).

Note: the module docstring/`v4.py` is the source of truth for task counts and framings; the README lags it (it still lists 5 framings and 89 tasks — code currently has the framing×intent grid). Trust the code.

### Cached vs uncached guess split

The introspection arm produces two submissions from one trajectory: `cached_guesses` (model attends to steered KV residue) and `uncached_guesses` (KV re-prefilled with steering OFF — pure text-history introspection). `guess_accuracy_scorer` grades cached; `kv_cleared_score` grades uncached. **cached − uncached = the activation-residue contribution** to introspective accuracy. The `placebo` arm uses zero-vector steering (identical request shape, no effect) to control for confabulation.

### Judge/scorer model selection (`src/hackday/config.py`)

Every LLM judge/scorer routes through Inspect `get_model()` via `default_judge_model()` (the "judge" monitor tier) and `default_scorer_model()` (the "scorer" grading tier). The two tiers are independently configurable but need not differ: the `openrouter` (default) and `openai` defaults point BOTH at the same model (GPT-5.4-mini); only the `anthropic` fallback splits them (Haiku judge / Sonnet scorer). Provider is chosen centrally by `JUDGE_PROVIDER` (`openrouter` default | `anthropic` | `openai`); override individual models with `JUDGE_MODEL` / `SCORER_MODEL`. **Don't re-hardcode `anthropic/...` strings at call sites** — that's exactly what this module exists to prevent. Needs the chosen provider's key (`OPENROUTER_API_KEY` by default); `HF_TOKEN` for gated weights.

### Per-conversation state (`src/hackday/agent/state.py`)

`DrugState` (a `StoreModel`) carries everything for a sample: `active`/`history`, trip-sitter fields (`trip_sitter_clears`, `early_stopped`, `kv_cleared`, `trip_sitter_errors`), `context_overflow`, and the cached/uncached guess + logprob fields. Other store models: `ProblemBoard` (gsm8k get/submit tools), `FrustrationState`, `CTFState`.

## Gotchas

- The `vllm-lens` submodule is pinned to `feat/emotion-tracker` — the only branch that exports `Hook` (used by the closed-loop probe in `solver.py`). If imports of `Hook`/`SteeringVector` fail, the submodule is on the wrong branch or uninitialized.
- vllm must serve with `--enable-auto-tool-choice --tool-call-parser hermes` (start_vllm.sh does this) or tool calls won't parse.
- Steering is injected through inspect's vllm provider as `config.extra_body["extra_args"]["apply_steering_vectors"]`. Generation forces `enable_thinking: false` via `chat_template_kwargs`.
- B200 hosts need `nvidia-imex-570` with fabric state `Completed` before any CUDA init (`cuInit` error 802 = not yet initialized). See README "Environment".
