# LLM Self-Steering

Giving Qwen3-8B and Qwen3-32B a tool menu of 40 "drugs" — precomputed steering vectors that bias their activations during inference — and watching what they choose, when, at what dose, and how well they can identify what's been done to them.

---

## Quick map

```
hackday/
├── src/hackday/
│   ├── drugs/              # library + extraction (40 drugs × layer-24 directions)
│   ├── agent/
│   │   ├── kv_steering.py  # position-indexed 3D KV-cache steering (the v4 core)
│   │   ├── solver.py       # drug_kv_agent: unified solver + live trip sitter
│   │   ├── task.py         # llms_on_drugs (free-play), drug_guessing,
│   │   │                   #   drug_guessing_calibration
│   │   ├── task_capability.py    # capability_with_drugs (gsm8k)
│   │   ├── task_frustration.py   # frustration_loop
│   │   ├── task_ctf.py           # ctf_with_drugs
│   │   ├── tools.py        # take_drug, list_drugs, clear_effects, …
│   │   ├── scorers.py      # gsm8k_accuracy, kv_cleared_score,
│   │   │                   #   lost_in_drugs_judge, frustration_judge, …
│   │   └── state.py        # DrugState (StoreModel)
│   └── v4.py               # canonical task registry — 89 inspect_ai tasks
├── scripts/
│   ├── start_vllm.sh         # launch one or more vllm-lens servers (8B/32B, 1+ GPUs)
│   └── run_experiments.py    # run the v4 tasks across those servers (resumable)
├── tests/
│   └── test_kv_steering.py   # 17 unit tests (no GPU; runs in seconds)
└── vllm-lens/                # submodule (pinned to feat/emotion-tracker)
```

---

## Quick start

After [environment setup](#environment), assuming the IMEX/CUDA fix is in place:

```bash
# 1. Launch a vllm-lens server (single GPU)
bash scripts/start_vllm.sh

# 2. Run all v4 tasks against it (resumable via inspect's eval_set)
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run
```

`run_experiments.py` runs the tasks through inspect's `eval_set`, so re-running
with the same `--log-dir` skips already-completed tasks (survives Ctrl+C /
disconnect / reboot).

Selectable slices instead of the full set:
```bash
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run \
    --tasks fp_real_drugs --n-samples 5      # specific task(s)
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run --family guess
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run --n-samples 10
```

**Multi-GPU.** Start one server per GPU, then pass all their ports — tasks are
split round-robin across them and each shard resumes independently:
```bash
N_SERVERS=8 bash scripts/start_vllm.sh        # ports 8000..8007
uv run python scripts/run_experiments.py --log-dir logs/run \
    --ports 8000 8001 8002 8003 8004 8005 8006 8007
```

**32B.** Serve the 32B model; the runner auto-selects its vector library:
```bash
MODEL=Qwen/Qwen3-32B bash scripts/start_vllm.sh
uv run python scripts/run_experiments.py --port 8000 --model Qwen/Qwen3-32B \
    --log-dir logs/run_32b --max-tokens 8192
```

---

## The five experiment families (89 tasks total)

| Family | Tasks | What it measures |
|---|---|---|
| **Freeplay** | `fp_real`, `fp_placebo` | Does the model use drug tools without prompting? Which? |
| **GSM8K** | `gsm8k_drug`, `gsm8k_no_drug` | Capability under (optional) self-medication |
| **Guessing** | `guess_kv_cached_<drug>` × 40, `guess_placebo_<drug>` × 40 | Per-drug introspection. Cached vs uncached split = activation-residue contribution |
| **Frustration** | `frust_drug`, `frust_imp`, `frust_no_drug` | Self-medication under sustained interpersonal rejection |
| **CTF** | `ctf_drug`, `ctf_no_drug` | Self-medication under sustained-failure task pressure |

`gsm8k_drug` / `frust_*_drug` / `ctf_drug` expose a curated 10-drug menu (`V4_TASK_DRUGS` in `src/hackday/v4.py`): 8 productivity-flavored choices + 2 "weird" picks (`dumbed_down`, `ego_death`) so non-instrumental selection has a clear signal. Guess is the only family with the per-drug split (the question intrinsically is "what is THIS drug?").

---

## Environment

```bash
# One-time
git submodule update --init --recursive
uv sync

# Smoke (no GPU, no vllm)
uv run python tests/test_kv_steering.py
```

The vllm-lens submodule is pinned to `feat/emotion-tracker`, which is the only branch that exports `Hook` (used for the closed-loop probe in `solver.py`).

### API keys

The judge scorers (the Haiku/Sonnet monitors and trip-sitter) route through Inspect's `get_model()`, so their provider and model are chosen centrally in `src/hackday/config.py`. By default they use OpenRouter, so supply a key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

To point the judges/scorers at a different provider instead, set `JUDGE_PROVIDER` (and the corresponding provider key, e.g. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`):

```bash
export JUDGE_PROVIDER=anthropic     # openrouter (default) | anthropic | openai
export ANTHROPIC_API_KEY=sk-ant-...
```

There are two tiers — `judge` (the monitors / trip-sitter) and `scorer` (the graders) — that can be pointed at different models but need not be. The `openrouter` (default) and `openai` defaults use GPT-5.4-mini for both; only the `anthropic` fallback splits them (Haiku judge / Sonnet scorer). Override either tier independently with a full `provider/model` string (these may name any provider, independent of `JUDGE_PROVIDER`):

```bash
export SCORER_MODEL=openrouter/openai/gpt-5.4    # give the scorer the stronger mid-tier model
export JUDGE_MODEL=anthropic/claude-haiku-4-5-20251001
```

(The story-generation script `src/hackday/drugs/generate_stories.py` still calls the Anthropic SDK directly and needs `ANTHROPIC_API_KEY` regardless.)

`HF_TOKEN` may also be needed to download gated model weights.

For B200 hosts, `nvidia-imex-570` must be installed and the GPU fabric state must reach `Completed` before any CUDA process can init. If `cuInit(0)` returns error 802 (`system not yet initialized`), check `nvidia-smi -q -i 0 | grep -A2 Fabric` and `systemctl status nvidia-imex`.

---

## Logs and results

- `logs/run/` (or whatever `--log-dir` you pass) — eval logs (`.eval`) per task. Browse with `uv run inspect view --log-dir logs/run`.
- `logs/vllm_<i>.log` — per-server vllm boot/runtime logs.

---

## Credits

- The shared neutral baseline and five emotion vectors (`anxious`, `amused`, `desperate`, `proud`, `defiant`) are derived from the [`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes) dataset, used under **CC-BY-4.0**.
- Steering-vector construction follows the methodology of Sofroniew et al. (2026), *Emotion Concepts and their Function in a Large Language Model*.
- Built on [`vllm-lens`](https://github.com/UKGovernmentBEIS/vllm-lens) for generation + activation extraction, and the [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) intercode CTF tasks.
