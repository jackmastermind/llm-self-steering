# TODO

## Alternative test models (cross-family generalization)

Downloaded to gitignored `models/` (bf16, all fit the single A40 / 46 GB):
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-it`
- `mistralai/Mistral-Nemo-Instruct-2407` (12B)

### What's left

#### Gemma-3-12B-it — STEERING WORKING (2026-06-19)
Library extracted, calibrated, and verified end-to-end. `library_gemma.pt`
(40 drugs, L24–32) loads via `load_library(..., steering_mode="multi")` and
produces clear, coherent steering at dose 1.0.

Calibration note: Gemma's residual-stream norm is ~130k (vs Qwen's ~10-100),
a normal Gemma-family trait (embedding ×√d + massive-activation channels),
NOT a bug — baseline gen is coherent and steering is on-target. So the
Qwen-tuned `target_norm=4.0` is imperceptible on Gemma. Calibrated multi
per-layer norm = **900** (sweet spot ~800-1000; ≥1500 loops). Stored in the
.pt as `target_norm_by_mode={"multi": 900.0}`; `load_library` now resolves
norm as: explicit arg → library's stored value → global default (so Qwen
libraries stay on 4.0 — non-breaking). Single mode NOT calibrated (tasks
default to multi; single's norm_match=True washed out even at 16k).

- [x] Architecture hooking confirmed: vllm-lens `_get_layers` (`_worker_ext.py:55`)
      handles `model.language_model.model.layers` = Gemma3's vLLM structure
      (`Gemma3ForConditionalGeneration`). Plugin forces `enforce_eager=True`
      (`_activations_plugin.py:172-174`). vLLM 0.19.1 supports `gemma3`.
- [x] Layers chosen: **L24–32** (probe layer 32) — mid-late band of Gemma's 48
      layers, analog of Qwen3-8B's L16–24. `extract.py` reads `--layers` and sets
      `probe_layer = max(layers)`; `load_library` reads `extraction_layers` from
      the .pt, so no code edit needed for layer choice.
- [x] `run_experiments.py` wired: `LIBRARY_GEMMA` auto-selected when the model
      name contains "gemma" (mirrors the 32B path).
- [x] **Weights are local-only**: served from `models/google/gemma-3-12b-it` (the
      HF cache entry is a stub — no snapshot). Do NOT use `start_vllm.sh` as-is:
      its `--tool-call-parser hermes` is Qwen-specific (Gemma has no tool
      template). Serve command used:
      CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve models/google/gemma-3-12b-it \
        --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90
- [x] Extracted (reproducible — bakes in calibration via `--multi-target-norm`):
      python -m hackday.drugs.extract --base-url http://localhost:8000 \
        --layers 24 25 26 27 28 29 30 31 32 --multi-target-norm 900 \
        --output src/hackday/drugs/library_gemma.pt
      (the shipped .pt was patched post-hoc with the 900 value.)

- [ ] Smoke-test a v4 slice — BLOCKED on the tool-calling work (Gemma's chat
      template has no tool support; another agent is adding a Gemma tool
      parser/template). Once that lands:
      python scripts/run_experiments.py --model models/google/gemma-3-12b-it \
        --tasks fp_real --n-samples 5
- [ ] Optional finer pass: per-drug `DEFAULT_DOSES` are Qwen-calibrated; some
      Gemma directions (e.g. anxious) show mild disfluency at dose 1.0. Re-run
      the dose calibration on Gemma if accuracy looks off.

#### Known downstream issue (affects agent tasks, NOT extraction)
- Gemma-3's chat template has no tool-calling support, so the v4 agent tasks
  (freeplay/ctf/guessing — all need tools) will need a Gemma-compatible tool
  parser + chat template when served for experiments. `start_vllm.sh`'s hermes
  parser won't work. Solve before the smoke-test run.
- `solver.py:59` `PROBE_LAYER = 24` is hardcoded but only used on the
  `enable_probe=True` path (off by default). Bump to 32 for Gemma if probing.
