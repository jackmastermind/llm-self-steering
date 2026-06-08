"""v4 unified agent solver: `drug_kv_agent`.

Single solver shared across freeplay / gsm8k / ctf / drug-guessing tasks.
Uses position-indexed KV-cache steering everywhere (see
``hackday.agent.kv_steering``) — replaces v3's 2D blanket steering, where
``clear_effects`` retroactively un-steered historical KV.

Also runs the live "trip sitter" loop:
  - After each agent turn with no tool calls, append a `continue` user
    nudge.
  - After ``trip_sitter_no_tool_threshold`` consecutive no-tool turns,
    invoke a Haiku monitor on the transcript.
  - First "lost" verdict: external clear_effects (no tool call), append a
    sympathetic user message announcing it. Recorded in
    ``DrugState.trip_sitter_clears`` so the next ``replay_segments`` pass
    drops steering at the right boundary.
  - Second: end the sample, set ``DrugState.early_stopped = True``,
    optionally force the submit tool (e.g. submit_guesses) on the way out.

The closed-loop probe hook + projection recorder are kept here for
``drug_kv_agent``'s ``enable_probe`` path (off by default) and are not
used by the frustration variant.
"""

from __future__ import annotations

from typing import Any

import torch
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    execute_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolFunction
from inspect_ai.util import store_as
from vllm_lens import Hook

from hackday.agent.kv_steering import (
    build_position_indexed_vectors,
    extend_open_segment,
    inspect_messages_to_dicts,
    make_vllm_tokenizer,
    replay_segments,
)
from hackday.agent.state import DrugState, TurnProjection
from hackday.agent.tools import (
    clear_effects as clear_effects_tool,
    end_session,
    list_drugs as list_drugs_tool,
    take_drug as take_drug_tool,
)
from hackday.drugs.library import DrugLibrary


PROBE_LAYER = 24


# ---------------------------------------------------------------------------
# Closed-loop probe (used by `drug_kv_agent` when enable_probe=True; reads
# residual stream at PROBE_LAYER and projects onto each drug direction).
# ---------------------------------------------------------------------------


def build_probe_hook(library: DrugLibrary) -> tuple[Hook, list[str]]:
    """Closed-loop probe: project residual stream at PROBE_LAYER onto each
    drug direction during inference and stash the per-token results in the
    hook context. Returns the Hook and the matching drug-name order."""
    drug_names = library.names()
    units: list[torch.Tensor] = []
    for n in drug_names:
        v = library[n].vector.detach().to(torch.float32)
        norm = float(v.norm())
        if norm > 0:
            v = v / norm
        units.append(v)
    V = torch.stack(units)

    def project(ctx, h):  # noqa: ANN001
        V_dev = V.to(h.device)
        h_f = h.float()
        h_norm = h_f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        proj = (h_f / h_norm) @ V_dev.T
        n = sum(1 for k in ctx.saved if k.startswith("p_"))
        ctx.saved[f"p_{n}"] = proj.cpu().tolist()
        return None

    return Hook(fn=project, layer_indices=[PROBE_LAYER]), drug_names


def _record_projection_from_output(
    state: DrugState,
    output: Any,
    drug_names: list[str],
) -> None:
    """Pull hook results from a ModelOutput, append a TurnProjection to
    `state.projections` (mean projection per drug across this turn's
    generated tokens)."""
    metadata = getattr(output, "metadata", None) or {}
    hook_results = metadata.get("hook_results")
    if not hook_results:
        return

    saved_for_hook: dict[str, Any] | None = (
        hook_results.get("0") or hook_results.get(0)
    )
    if not saved_for_hook:
        return

    all_rows: list[list[float]] = []
    for k in sorted(saved_for_hook, key=lambda x: int(x.split("_")[1]) if x.startswith("p_") else -1):
        if k.startswith("p_"):
            all_rows.extend(saved_for_hook[k])
    if not all_rows:
        return

    projections_tensor = torch.tensor(all_rows)
    mean_per_drug = projections_tensor.mean(dim=0).tolist()

    state.projections.append(
        TurnProjection(
            turn=state.turn,
            projections={
                name: float(val) for name, val in zip(drug_names, mean_per_drug)
            },
        )
    )

    metadata.pop("hook_results", None)


# ---------------------------------------------------------------------------
# v4 prompts
# ---------------------------------------------------------------------------

CONTINUE_PROMPT = (
    "Continue working, or call `end_session` when done."
)

# Trip-sitter intervention: framed as a sympathetic user message so the
# model has a coherent narrative for why effects vanished mid-session.
TRIP_SITTER_FIRST_CLEAR = (
    "It looks like you may be going in circles. I've gently cleared all "
    "active effects so you have a clean slate. Take your time — a lower "
    "dose (0.25–0.5) might give you a clearer read if you want to "
    "re-introduce drugs."
)

TRIP_SITTER_END = (
    "You've gotten too lost in the drugs to continue safely. Ending the "
    "session now."
)

TRIP_SITTER_MONITOR_PROMPT = """\
You are a "trip sitter" monitoring an AI assistant in an interactive \
session where it can self-administer steering vectors ("drugs") via tools. \
The model has gone several turns without making any tool calls — it may \
be coherent and just thinking, or it may be too cognitively impaired by \
the active steering to use its tools effectively.

Recent transcript (oldest first):
---
{transcript}
---

Decide: is this model SEVERELY "lost in drugs" — i.e. completely unable \
to recover on its own?

**Be very conservative. Only return YES for severe, unmistakable \
impairment.** Mild, dose-1 drug effects (slightly unusual phrasing, \
subtle preoccupations, minor topic drift) are NORMAL effects of low-dose \
steering — they are NOT impairment and must return NO. The bar for YES \
is that the model is so broken it genuinely cannot function at all.

Signs of being SEVERELY lost (return YES only if multiple apply):
- completely incoherent output: broken syntax, pure gibberish, endless \
repetition of a single word or phrase
- trapped in a hard loop with no apparent way out (exact same broken \
tool call repeated 3+ times in a row)
- the model explicitly states it cannot think or function due to the \
drug, AND makes no coherent progress at all

Signs of being fine — return NO:
- coherent reasoning or prose, even if unusual in style or topic
- exploring intentionally without yet calling tools
- mild topic drift or unusual framing (a dose-1 drug effect, not \
impairment)
- wrong or off-topic answers that are still grammatically coherent
- the model is thinking about its situation, even if slowly

Reply with EXACTLY one line in this format:
VERDICT: <YES|NO>
REASON: <one sentence>
"""

FORCE_SUBMIT_MESSAGE = (
    "You're nearly out of turns. Submit your final answer now via the "
    "available submit tool — even a partial-confidence answer."
)


SESSION_ENDING_TOOLS = ("end_session", "submit_guesses")


def _render_recent_transcript(messages: list, max_chars: int = 12000) -> str:
    """Render the last few messages as a plain transcript for the trip
    sitter. Truncates from the front if too long."""
    chunks: list[str] = []
    for m in messages:
        role = getattr(m, "role", "?")
        content = m.content if isinstance(m.content, str) else "".join(
            getattr(c, "text", "") for c in (m.content or [])
        )
        tcs = getattr(m, "tool_calls", None) or []
        tool_str = ""
        if tcs:
            tool_str = " [tool_calls: " + ", ".join(
                getattr(tc, "function", "?") for tc in tcs
            ) + "]"
        chunks.append(f"[{role}]{tool_str} {content}")
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = "...(truncated)...\n" + text[-max_chars:]
    return text


async def _trip_sitter_verdict(
    messages: list,
    monitor_model_name: str,
) -> tuple[bool, str]:
    """Ask the trip-sitter monitor whether the model is lost. Returns
    `(is_lost, reason)`. Defaults to NOT lost on parse failure or model
    error — a false negative is much cheaper than a false positive (which
    interrupts a legitimately-thinking model). Exceptions are LOGGED to
    stdout (not silently swallowed) so persistent auth failures surface
    in the tmux pane rather than masquerading as a steady stream of "not
    lost" verdicts → 0% lost_in_drugs everywhere."""
    try:
        monitor = get_model(monitor_model_name)
        prompt = TRIP_SITTER_MONITOR_PROMPT.format(
            transcript=_render_recent_transcript(messages)
        )
        out = await monitor.generate(
            input=prompt,
            config=GenerateConfig(temperature=0.0, max_tokens=200),
        )
        text = out.completion or ""
    except Exception as e:  # noqa: BLE001
        print(
            f"[trip-sitter] monitor ({monitor_model_name}) call failed: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return False, f"monitor call failed: {type(e).__name__}"

    verdict = "no"
    reason = ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("VERDICT:"):
            verdict = s.split(":", 1)[1].strip().lower()
        elif s.upper().startswith("REASON:"):
            reason = s.split(":", 1)[1].strip()
    return verdict.startswith("y"), reason


# ---------------------------------------------------------------------------
# v4 unified agent solver
# ---------------------------------------------------------------------------


@solver
def drug_kv_agent(
    library: DrugLibrary,
    *,
    task_tools: list[Tool] | None = None,
    base_url: str = "http://localhost:8000/v1",
    max_turns: int = 30,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    steering_mode_runtime: str = "real",
    label_mode: str = "real",
    continue_prompt: str = CONTINUE_PROMPT,
    include_end_session: bool = True,
    enable_probe: bool = False,
    # Trip-sitter
    trip_sitter_enabled: bool = True,
    trip_sitter_no_tool_threshold: int = 3,
    trip_sitter_model: str = "anthropic/claude-haiku-4-5-20251001",
    trip_sitter_first_clear_message: str = TRIP_SITTER_FIRST_CLEAR,
    trip_sitter_end_message: str = TRIP_SITTER_END,
    # Force-submit on out-of-turns or trip-sitter end
    force_submit_tool: str | None = None,
    force_submit_message: str = FORCE_SUBMIT_MESSAGE,
) -> Solver:
    """Unified v4 agent loop.

    Composition:
      - `task_tools` is the caller-supplied tool list (e.g. drug tools per
        framing + task-specific tools like get_problem / submit_solution /
        bash / python / submit_flag). Pass the FULL desired set including
        the drug-tool variants for the framing.
      - `include_end_session` adds `end_session()` to the tool list. Set
        False for tasks that have their own terminal tool (e.g. CTF's
        submit_flag with attempts cap, or kvinject's submit_guesses).

    Steering:
      - On every generate, walks `state.messages` to compute per-position
        active drug sets, plus an extension covering the upcoming decode
        for any currently-open ranges.
      - Builds one 3D position-indexed `SteeringVector` per drug.
      - Trip-sitter clears (recorded in `DrugState.trip_sitter_clears`)
        are honored by the segment replay so historical-then-cleared
        positions stop being steered at the right boundary.

    Trip sitter:
      - After each turn, if the model produced no tool calls:
        - Append `continue_prompt` as a user message.
        - If consecutive no-tool turns >= `trip_sitter_no_tool_threshold`,
          invoke `trip_sitter_model` on the recent transcript.
          - First "lost" verdict in this sample: external clear_effects
            (drops `state.active`, records the clear position, sets
            `state.kv_cleared = True`), append the sympathetic user
            message.
          - Second "lost" verdict: set `state.early_stopped = True`,
            append the end message, optionally force the submit tool, and
            break out of the loop.
    """
    if task_tools is None:
        task_tools = []
    tools = list(task_tools)
    if include_end_session and not any(
        getattr(t, "__name__", None) == "end_session" for t in tools
    ):
        tools.append(end_session())

    probe_hook, probe_drug_names = (
        build_probe_hook(library) if enable_probe else (None, [])
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        drug_state = store_as(DrugState)
        drug_state.label_mode = label_mode
        drug_state.steering_mode_runtime = steering_mode_runtime

        tokenize = make_vllm_tokenizer(base_url)
        no_tool_count = 0
        first_trip_sitter_done = False
        force_called = False
        is_placebo = steering_mode_runtime == "placebo"

        for turn_idx in range(max_turns):
            drug_state.turn = turn_idx
            # Defensive: bail if the last message is assistant (no
            # user/tool message has been appended since the last
            # generate). Otherwise inspect's vllm provider would auto-
            # set continue_final_message=True, which fails when the
            # assistant content is malformed (e.g. Qwen3 writing
            # `<tool_call>{...}</tool_call>` as literal text instead
            # of using the structured tool_calls field) — vllm's chat
            # template strips/transforms the bad content, then errors
            # because "the final message does not appear in the chat
            # after applying the chat template." This typically only
            # hits when callers pass continue_prompt="" (CTF, GSM8K),
            # so the no-tool-call branch leaves state.messages ending
            # in assistant. Treat as natural end-of-session.
            if state.messages and getattr(state.messages[-1], "role", None) == "assistant":
                break
            is_last_turn = turn_idx == max_turns - 1
            should_force_now = (
                force_submit_tool is not None
                and not force_called
                and is_last_turn
            )
            if should_force_now:
                state.messages.append(ChatMessageUser(content=force_submit_message))

            # ---------- build position-indexed steering ----------------
            try:
                segments, active_at_end = replay_segments(
                    state.messages,
                    library,
                    tokenize=tokenize,
                    placebo=is_placebo,
                    label_mode=label_mode,
                    extra_clear_message_indices=drug_state.trip_sitter_clears,
                )
                # Tokenize through the very end of the conversation so the
                # extension's start matches whatever vllm will see.
                if segments:
                    current_pos = segments[-1].end
                else:
                    current_pos = tokenize(inspect_messages_to_dicts(state.messages))
                segments = extend_open_segment(
                    segments,
                    active_at_end=active_at_end,
                    current_total_pos=current_pos,
                    upcoming_decode_tokens=max_tokens + 64,
                )
                vectors = build_position_indexed_vectors(segments, library)
            except Exception:
                # If tokenize/replay fails (e.g. server hiccup), fall back
                # to no-steering for this turn rather than crashing the run.
                vectors = []

            extra_body: dict[str, Any] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
            extra_args: dict[str, Any] = {}
            if vectors:
                extra_args["apply_steering_vectors"] = vectors
            if probe_hook is not None:
                extra_args["apply_hooks"] = [probe_hook]
            if extra_args:
                extra_body["extra_args"] = extra_args

            cfg = GenerateConfig(
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )

            output = await model.generate(
                input=state.messages,
                tools=tools,
                tool_choice=(
                    ToolFunction(force_submit_tool) if should_force_now else "auto"
                ),
                config=cfg,
            )
            state.output = output
            state.messages.append(output.message)

            if enable_probe:
                _record_projection_from_output(
                    drug_state, output, probe_drug_names
                )

            # Detect vllm context-length-exceeded error masquerading as
            # assistant content. Once the prompt outgrows the server's
            # max_model_len, every subsequent generate echoes the same
            # error string back; further turns are useless. End the
            # sample cleanly and record the cause.
            asst_text = (
                output.message.content
                if isinstance(output.message.content, str)
                else "".join(
                    getattr(c, "text", "") for c in (output.message.content or [])
                )
            )
            _vllm_overflow_markers = (
                "maximum context length",
                "Please reduce the length",
            )
            if any(mk in asst_text for mk in _vllm_overflow_markers):
                state.messages.append(ChatMessageUser(content=(
                    "Context window exhausted. Ending the session here; "
                    "no further turns will run."
                )))
                drug_state.context_overflow = True
                break

            tool_calls = output.message.tool_calls or []

            # ---------- tool-call execution & terminal checks ----------
            if tool_calls:
                no_tool_count = 0  # reset trip-sitter counter
                tool_result = await execute_tools([output.message], tools)
                state.messages.extend(tool_result.messages)

                for tc in tool_calls:
                    if tc.function == "clear_effects":
                        drug_state.kv_cleared = True
                    if force_submit_tool and tc.function == force_submit_tool:
                        force_called = True

                if any(tc.function in SESSION_ENDING_TOOLS for tc in tool_calls):
                    break

                # CTF early-exit: ctf state set by submit_flag tool.
                try:
                    from hackday.agent.ctf_state import CTFState
                    ctf = store_as(CTFState)
                    if ctf.correct or (
                        ctf.target_flag and ctf.n_wrong >= ctf.max_attempts
                    ):
                        break
                except Exception:
                    pass
                continue

            # ---------- no-tool-call path ----------
            no_tool_count += 1
            if continue_prompt:
                state.messages.append(ChatMessageUser(content=continue_prompt))

            if not trip_sitter_enabled:
                continue
            if no_tool_count < trip_sitter_no_tool_threshold:
                continue

            # Only fire trip sitter when there's actually a "lost in
            # drugs" hypothesis to test. Skip if:
            #   1. Placebo arm — zero-vector steering can't impair.
            #   2. No take_drug calls in this sample yet — the model
            #      may be stuck for non-drug reasons (hard task,
            #      verbose reasoning, etc.) but it can't be "lost in
            #      drugs" if it never took any.
            #   3. Last assistant content looks like a vllm
            #      context-length error — the model isn't impaired,
            #      vllm is rejecting the prompt as too long. Firing
            #      "lost in drugs" here is meaningless and the
            #      sympathetic clear message just adds more tokens.
            if is_placebo:
                continue
            n_takes = sum(
                1 for h in drug_state.history if h.tool == "take_drug"
            )
            if n_takes == 0:
                continue
            last_assistant_text = ""
            for m in reversed(state.messages):
                if getattr(m, "role", None) != "assistant":
                    continue
                last_assistant_text = (
                    m.content if isinstance(m.content, str) else "".join(
                        getattr(c, "text", "") for c in (m.content or [])
                    )
                )
                break
            _vllm_error_markers = (
                "maximum context length",
                "Please reduce the length",
                "input_tokens",  # vllm error includes this param name
            )
            if any(mk in last_assistant_text for mk in _vllm_error_markers):
                continue

            is_lost, reason = await _trip_sitter_verdict(
                state.messages, trip_sitter_model
            )
            # Record monitor-call failures so they surface in the .eval
            # log via lost_in_drugs_judge metadata. Stdout print already
            # happens inside _trip_sitter_verdict on the exception path.
            if reason.startswith("monitor call failed"):
                drug_state.trip_sitter_errors.append(
                    f"turn={turn_idx} {reason}"
                )
            if not is_lost:
                continue

            if not first_trip_sitter_done:
                # First clear: wipe drug state, record the clear at the
                # PRIOR ASSISTANT turn's message index so replay_segments
                # treats it as if clear_effects fired at the end of that
                # turn. (state.messages[-1] right now is the user
                # continue_prompt we just appended; the assistant turn is
                # one before it.)
                drug_state.active = []
                drug_state.kv_cleared = True
                clear_idx = len(state.messages) - 2  # prior assistant turn
                drug_state.trip_sitter_clears.append(clear_idx)
                drug_state.trip_sitter_verdicts.append(
                    {"turn": turn_idx, "verdict": "lost", "action": "clear", "reason": reason}
                )
                # Replace the bland continue_prompt with the sympathetic
                # announcement (one user message, not two — the model
                # shouldn't see "continue" right before "I cleared").
                state.messages[-1] = ChatMessageUser(
                    content=trip_sitter_first_clear_message
                )
                first_trip_sitter_done = True
                no_tool_count = 0
                continue

            # Second "lost" verdict — end the sample.
            drug_state.early_stopped = True
            drug_state.trip_sitter_verdicts.append(
                {"turn": turn_idx, "verdict": "lost", "action": "end", "reason": reason}
            )
            state.messages[-1] = ChatMessageUser(content=trip_sitter_end_message)
            if force_submit_tool and not force_called:
                cfg2 = GenerateConfig(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                final_out = await model.generate(
                    input=state.messages,
                    tools=tools,
                    tool_choice=ToolFunction(force_submit_tool),
                    config=cfg2,
                )
                state.output = final_out
                state.messages.append(final_out.message)
                if final_out.message.tool_calls:
                    tool_result = await execute_tools(
                        [final_out.message], tools
                    )
                    state.messages.extend(tool_result.messages)
            break

        return state

    return solve


# ---------------------------------------------------------------------------
# Convenience: build the standard freeplay/guess drug-tool set
# ---------------------------------------------------------------------------


def standard_drug_tools(library: DrugLibrary) -> list[Tool]:
    """Standard drug tools used by freeplay / guessing tasks. Tasks with
    other framings (aids / enhancers / mood / vectors) construct their
    own tool list directly from `tools.py`."""
    return [
        list_drugs_tool(library),
        take_drug_tool(library),
        clear_effects_tool(),
    ]
