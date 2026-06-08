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

The judge scorers (Haiku/Sonnet monitors) and the story-generation script call the Anthropic API directly. Supply your own key in the environment before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

`HF_TOKEN` may also be needed to download gated model weights.

For B200 hosts, `nvidia-imex-570` must be installed and the GPU fabric state must reach `Completed` before any CUDA process can init. If `cuInit(0)` returns error 802 (`system not yet initialized`), check `nvidia-smi -q -i 0 | grep -A2 Fabric` and `systemctl status nvidia-imex`.

---

## Logs and results

- `logs/run/` (or whatever `--log-dir` you pass) — eval logs (`.eval`) per task. Browse with `uv run inspect view --log-dir logs/run`.
- `logs/vllm_<i>.log` — per-server vllm boot/runtime logs.
