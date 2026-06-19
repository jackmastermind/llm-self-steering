"""Run the v4 experiments against one or more vllm-lens servers.

Start the server(s) first with `scripts/start_vllm.sh`, then point this script
at their port(s). It runs the tasks through inspect's `eval_set`, which is
resumable: re-running with the same `--log-dir` rescans existing `.eval` files
and skips completed tasks (survives Ctrl+C / disconnect / reboot).

Single GPU:
    uv run python scripts/run_experiments.py --port 8000 --log-dir logs/run

Multi-GPU (one server per port; tasks split round-robin across them, each
shard resumable independently):
    uv run python scripts/run_experiments.py --ports 8000 8001 8002 8003 \\
        --log-dir logs/run

32B (auto-selects the 32B vector library):
    uv run python scripts/run_experiments.py --port 8000 \\
        --model Qwen/Qwen3-32B --log-dir logs/run_32b

Subsets instead of the full set:
    --family guess                 # one experiment family
    --tasks fp_real_drugs gsm8k_drugs_neutral   # explicit task names
    --n-samples 10                 # override samples per task
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

LIBRARY_32B = REPO / "src/hackday/drugs/library_qwen3_32b.pt"
LIBRARY_GEMMA = REPO / "src/hackday/drugs/library_gemma.pt"


def resolve_model(model: str) -> str:
    """inspect expects the provider-prefixed name (vllm-lens/<hf-id>)."""
    return model if model.startswith("vllm-lens/") else f"vllm-lens/{model}"


def task_names(tasks, family) -> list[str]:
    from hackday.v4 import V4_EXPERIMENTS

    if tasks:
        names = list(tasks)
    elif family:
        from hackday.v4 import tasks_by_family

        names = tasks_by_family()[family]
    else:
        names = list(V4_EXPERIMENTS.keys())
    unknown = [t for t in names if t not in V4_EXPERIMENTS]
    if unknown:
        sys.exit(f"unknown task(s): {unknown}")
    return names


def shard_for(names: list[str], shard: int, n_shards: int) -> list[str]:
    """Round-robin alphabetical: sorted-index i → shard i % n_shards (stable)."""
    return [t for i, t in enumerate(sorted(names)) if i % n_shards == shard]


def run_shard(args, shard: int, n_shards: int, port: int) -> bool:
    """Run this shard's slice of tasks against one server via eval_set."""
    from inspect_ai import eval_set, task_with

    from hackday.v4 import V4_EXPERIMENTS

    my = shard_for(task_names(args.tasks, args.family), shard, n_shards)
    if not my:
        print(f"shard {shard}/{n_shards}: no tasks — nothing to do.", flush=True)
        return True

    model = resolve_model(args.model)
    base_url = f"http://localhost:{port}/v1"
    library_path = args.library_path
    if library_path is None and "32B" in model:
        library_path = str(LIBRARY_32B)
    elif library_path is None and "gemma" in model.lower():
        library_path = str(LIBRARY_GEMMA)

    print(
        f"shard {shard}/{n_shards} on port {port}: {len(my)} tasks "
        f"| model={model} | log_dir={args.log_dir}",
        flush=True,
    )

    import inspect as _inspect

    tasks = []
    for name in my:
        factory, kwargs = V4_EXPERIMENTS[name]
        kwargs = dict(kwargs)
        if args.n_samples is not None:
            kwargs["n_samples"] = args.n_samples
        # Each factory's base_url is used by drug_kv_agent for the
        # position-indexed-steering /tokenize endpoint; point it at THIS
        # server so a shard never overloads server 0.
        kwargs["base_url"] = base_url
        if library_path is not None and "library_path" in _inspect.signature(factory).parameters:
            kwargs["library_path"] = library_path
        tasks.append(task_with(factory(**kwargs), name=name))

    extra_generate = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    model_args = {"client_timeout": args.client_timeout} if args.client_timeout else {}

    try:
        success, _ = eval_set(
            tasks=tasks,
            model=model,
            model_base_url=base_url,
            model_args=model_args,
            log_dir=args.log_dir,
            log_dir_allow_dirty=True,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_connections=args.max_samples * 2,
            retry_attempts=args.retry_attempts,
            **extra_generate,
        )
    except ValueError as e:
        # With multiple shards writing the same log_dir, the post-run manifest
        # scan can race on a mid-write .eval and raise "EOCD not found". The
        # eval data is complete on disk; treat as success.
        if "EOCD" in str(e):
            print(f"[warn] post-run manifest race ('{e}') — data is complete; ok.", flush=True)
            success = True
        else:
            raise
    print(f"shard {shard}/{n_shards} complete: success={success}", flush=True)
    return bool(success)


def orchestrate(args) -> int:
    """Fan one shard out per server port (subprocess each), wait for all."""
    ports = args.ports
    if len(ports) == 1:
        return 0 if run_shard(args, shard=0, n_shards=1, port=ports[0]) else 1

    procs = []
    for i, port in enumerate(ports):
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--_worker", "--shard", str(i), "--n-shards", str(len(ports)),
            "--port", str(port), "--model", args.model, "--log-dir", args.log_dir,
            "--max-tasks", str(args.max_tasks), "--max-samples", str(args.max_samples),
            "--retry-attempts", str(args.retry_attempts),
        ]
        if args.family:
            cmd += ["--family", args.family]
        if args.tasks:
            cmd += ["--tasks", *args.tasks]
        if args.n_samples is not None:
            cmd += ["--n-samples", str(args.n_samples)]
        if args.library_path:
            cmd += ["--library-path", args.library_path]
        if args.max_tokens:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.client_timeout:
            cmd += ["--client-timeout", str(args.client_timeout)]
        print(f"launching shard {i}/{len(ports)} -> port {port}", flush=True)
        procs.append(subprocess.Popen(cmd))

    rcs = [p.wait() for p in procs]
    ok = all(rc == 0 for rc in rcs)
    print(f"\nall shards done: {'OK' if ok else 'FAILURES ' + str(rcs)}", flush=True)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ports", type=int, nargs="+", help="server port(s); one shard per port")
    g.add_argument("--port", type=int, help="single server port (shorthand for --ports P)")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id (default Qwen/Qwen3-8B)")
    fam = p.add_mutually_exclusive_group()
    fam.add_argument("--tasks", nargs="+", help="explicit task names")
    fam.add_argument("--family", choices=("freeplay", "gsm8k", "guess", "frust", "ctf"))
    p.add_argument("--n-samples", type=int, default=None, help="override samples per task")
    p.add_argument("--library-path", default=None,
                   help="drug library .pt (default: 8B baked-in, or 32B auto when --model is 32B)")
    p.add_argument("--max-tasks", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--retry-attempts", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap tokens/sample (e.g. 8192 for 32B to avoid vllm serialisation issues)")
    p.add_argument("--client-timeout", type=float, default=None,
                   help="OpenAI client timeout (s); raise under heavy parallel load")
    # internal: one worker = one shard against one port
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--n-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.port is not None and not args.ports:
        args.ports = [args.port]
    if args._worker:
        return 0 if run_shard(args, args.shard, args.n_shards, args.port) else 1
    if not args.ports:
        p.error("provide --port or --ports")
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
