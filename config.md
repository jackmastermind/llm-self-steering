# Machine compatibility report — `llm-self-steering`

Freshly cloned AISI repo, audited against **this** host on 2026-06-19.
This documents what must change before experiments can run here.

## TL;DR

This machine is a **single NVIDIA A40 (46 GB, Ampere sm_86)** in a **non-systemd
container**. The repo was authored for a **multi-GPU B200 host**. None of the
B200/IMEX setup applies here, but several real blockers must be fixed first:

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | `vllm-lens/` submodule not initialized (empty dir) | **Blocker** | `git submodule update --init --recursive` |
| 2 | No `.venv` yet | **Blocker** | `uv sync` |
| 3 | `tmux` not installed (required by `start_vllm.sh`) | **Blocker** for the launch script | install tmux, or launch vllm manually |
| 4 | `ANTHROPIC_API_KEY` unset (judge scorers, CTF, story gen) | **Blocker** for scored runs | `export ANTHROPIC_API_KEY=...` |
| 5 | `docker` not installed | **Blocker for CTF family only** | install docker, or skip `ctf_*` tasks |
| 6 | Single GPU → 32B model and multi-GPU examples won't run | Constraint | use Qwen3-8B, single server only |
| 7 | README B200/IMEX instructions don't apply | Documentation | ignore; see note below |

---

## Detected host environment

| Component | This machine | Repo expectation |
|-----------|-------------|------------------|
| GPU | 1× **A40**, 46 GB, Ampere (sm_86) | multi-GPU (README shows 8 servers/8 GPUs); B200 mentioned |
| NVIDIA driver | 570.211.01 (CUDA 12.8 capable) | — |
| CUDA toolkit | 12.4 in `/usr/local/cuda-12.4`; **no `nvcc`** | torch wheels pinned to `cu126` |
| GPU Fabric state | `N/A` (normal for A40) | README warns about B200 IMEX/fabric `Completed` state |
| Init system | **not systemd** (`systemctl` unavailable) | README uses `systemctl status nvidia-imex` |
| Python on PATH | 3.11.15 (conda `arena-env`) | `requires-python >=3.12`, `.python-version` = 3.12 |
| uv | 0.11.17 ✓ | `uv_build >=0.11.7,<0.12.0` ✓ |
| Disk (`/`) | 200 GB, 197 GB free ✓ | model weights fit easily |
| `HF_TOKEN` | set ✓ | needed for gated weights (Qwen3 is ungated, so optional) |
| `ANTHROPIC_API_KEY` | **unset** ✗ | required by judge scorers / CTF / story gen |
| `tmux` | **missing** ✗ | required by `scripts/start_vllm.sh` |
| `docker` | **missing** ✗ | required by CTF sandbox |
| `curl` | present ✓ | used by `start_vllm.sh` readiness poll |

Pinned versions in `uv.lock`: torch 2.10.0, vllm 0.19.1, transformers 5.5.4,
inspect-ai 0.3.209. vllm 0.19.1 + torch 2.10 support Ampere (sm_86), so the A40
is fine — torch wheels bundle their own CUDA runtime, so the missing system
`nvcc` / older 12.4 toolkit is not a problem (driver 12.8 covers the cu126
runtime).

---

## What must change, in order

### 1. Initialize the submodule (hard blocker)
`vllm-lens/` is empty. `pyproject.toml` declares it as an editable path
dependency (`vllm-lens = { path = "./vllm-lens", editable = true }`), so
`uv sync` fails outright until it's populated.

```bash
git submodule update --init --recursive
```
This checks out the pinned commit on `feat/emotion-tracker` (the only branch that
exports `Hook`, used by `solver.py`'s closed-loop probe).

### 2. Create the environment
```bash
uv sync   # uv auto-fetches a managed Python 3.12; the conda 3.11 on PATH is not used
```
The conda `arena-env` Python (3.11) does **not** satisfy `requires-python >=3.12`,
but `uv` provisions its own interpreter, so no manual Python install is needed.

### 3. Install `tmux` (or bypass the launch script)
`scripts/start_vllm.sh` runs each vllm server inside a tmux session and will fail
immediately without it.
```bash
apt-get update && apt-get install -y tmux
```
Alternatively, launch vllm directly (no tmux) for a single GPU:
```bash
uv run vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90
```

### 4. Export the Anthropic key
The judge scorers (`lost_in_drugs_judge`, `frustration_judge`, Haiku/Sonnet
monitors) and the story-generation script call the Anthropic API. Without it,
those tasks' scoring will fail.
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Stay on Qwen3-8B, single server (hardware constraint)
With one 46 GB A40:
- **Qwen3-8B fits** comfortably (bf16 ≈ 16 GB + KV cache). This is the default
  and the supported path here.
- **Qwen3-32B will not fit** on a single A40 in bf16 (≈ 64 GB needed). The
  `MODEL=Qwen/Qwen3-32B` examples require either multiple GPUs (TP) we don't have,
  or quantization that isn't configured. **Skip 32B on this host.**
- The **multi-GPU examples** in the README (`N_SERVERS=8`, `--ports 8000..8007`)
  are not runnable — only one GPU. Use `--port 8000` with a single server.

So the working run on this machine is:
```bash
bash scripts/start_vllm.sh                      # 1× Qwen3-8B on port 8000 (needs tmux)
uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run
```

### 6. CTF family needs Docker (skip otherwise)
`ctf_with_drugs` (`ctf_drug`, `ctf_no_drug`) builds an inspect sandbox via
`docker compose` and needs outbound network for picoCTF artifacts. Docker is not
installed. Either install Docker, or simply don't run the `ctf_*` tasks — the
other four families (freeplay, gsm8k, guess, frust) don't need it. Use
`--tasks`/`--family` to exclude CTF.

---

## What does NOT need changing (don't waste time here)

- **IMEX / GPU fabric / `nvidia-imex` / `cuInit` error 802.** This is B200-only.
  The A40 has no NVLink fabric; `Fabric State: N/A` is the correct healthy
  state. The README "assuming the IMEX/CUDA fix is in place" precondition and the
  `systemctl status nvidia-imex` check simply don't apply on this host (and
  `systemctl` won't work anyway — no systemd). No action required.
- **CUDA toolkit version / missing `nvcc`.** torch/vllm ship prebuilt wheels with
  bundled CUDA; the driver (12.8) supports the cu126 runtime. No system CUDA
  install needed.
- **`HF_TOKEN`.** Already set, and Qwen3 weights are ungated anyway.

---

## Smoke test (no GPU)

Before any GPU run, confirm the env is wired up:
```bash
uv run python tests/test_kv_steering.py   # 17 unit tests, no GPU/vllm, runs in seconds
```

## Suggested bring-up sequence

```bash
git submodule update --init --recursive
uv sync
apt-get update && apt-get install -y tmux
export ANTHROPIC_API_KEY=sk-ant-...
uv run python tests/test_kv_steering.py                          # smoke
bash scripts/start_vllm.sh                                       # Qwen3-8B, port 8000
uv run python scripts/run_experiments.py --port 8000 \
    --log-dir logs/run --family guess --n-samples 5              # small first slice
```
