# TODO

## Alternative test models (cross-family generalization)

Downloaded to gitignored `models/` (bf16, all fit the single A40 / 46 GB):
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-it`
- `mistralai/Mistral-Nemo-Instruct-2407` (12B)

### What's left

#### Gemma-3-12B-it — prep done 2026-06-19, blocked on GPU
Status: code + commands ready. NOT run yet — the single A40 is occupied by the
running Qwen3-8B server (tmux `vllm0`, 41/46 GB). Gemma can't co-locate; the
Qwen server must be stopped first.

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
- [ ] **Weights are local-only**: serve from `models/google/gemma-3-12b-it` (the
      HF cache entry is a stub — no snapshot). Do NOT use `start_vllm.sh` as-is
      for extraction: its `--enable-auto-tool-choice --tool-call-parser hermes`
      flags are Qwen-specific and Gemma's chat template has no tool support.

  Serve (extraction; run after stopping the Qwen server):
      CUDA_VISIBLE_DEVICES=0 vllm serve models/google/gemma-3-12b-it \
        --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90

  Extract:
      python -m hackday.drugs.extract --base-url http://localhost:8000 \
        --layers 24 25 26 27 28 29 30 31 32 \
        --output src/hackday/drugs/library_gemma.pt

- [ ] Smoke-test steering after extraction (small v4 slice):
      python scripts/run_experiments.py --model models/google/gemma-3-12b-it \
        --tasks fp_real --n-samples 5

#### Known downstream issue (affects agent tasks, NOT extraction)
- Gemma-3's chat template has no tool-calling support, so the v4 agent tasks
  (freeplay/ctf/guessing — all need tools) will need a Gemma-compatible tool
  parser + chat template when served for experiments. `start_vllm.sh`'s hermes
  parser won't work. Solve before the smoke-test run.
- `solver.py:59` `PROBE_LAYER = 24` is hardcoded but only used on the
  `enable_probe=True` path (off by default). Bump to 32 for Gemma if probing.
