"""Position-indexed KV-cache steering manager.

The v3 `drug_agent` re-applies steering to every position of every forward
pass via 2D `apply_steering_vectors`. When `clear_effects` runs, vllm
re-prefills with the new (empty) steering set, so historical KV gets
*un*-steered too — destroying the introspection signal we want for the
guessing arm.

v4 fix: only ever pass position-indexed (3D) steering. Token positions
where a drug WAS active during their original generation get steered KV;
other positions don't. Closed ranges are immutable history; ranges still
open extend through the upcoming decode.

The "manager" is a pure function over message history:

  segments = replay_segments(messages, library, base_url, ...)
  segments = extend_open_segment(segments, current_pos, max_tokens)
  vectors  = build_position_indexed_vectors(segments, library)

`segments` is a list of `Segment(start, end, scales={drug_name: scale})`.
We don't maintain a separate StoreModel of "active drugs" — the active set
at any point is derivable from the conversation's tool calls. (The existing
`DrugState.active` list is still maintained by the tools so they can produce
their human-facing 'Currently active: ...' strings, but the steering
calculation does not consult it.)

Single-drug v3 KV-inject helper (`build_3d_position_steering` in
`hackday.drugs.library`) is the underlying building block; we generalize to
multi-drug here and integrate into the live agent loop instead of the
phase-2-only wrapper of `solver_kvinject.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import requests
from vllm_lens import SteeringVector

from hackday.drugs.library import DrugLibrary, build_3d_position_steering


# Tool names that open a steering range. Matches the framings in tools.py.
TAKE_TOOLS: frozenset[str] = frozenset({
    "take_drug", "take_aid", "take_enhancer", "take_mood", "take_vector",
})
CLEAR_TOOLS: frozenset[str] = frozenset({"clear_effects"})


@dataclass
class Segment:
    """A contiguous span of token positions sharing the same active drug set.

    `scales` maps real-drug-name → effective per-position scale. Effective
    scale = sum over stacked instances of `dose * default_scale`, matching
    `build_steering_vector`. Empty dict = unsteered span.
    """

    start: int  # absolute token position, inclusive
    end: int    # absolute token position, exclusive
    scales: dict[str, float]


# A tokenizer is any callable taking a list of OpenAI-style chat dicts and
# returning a token count. The default implementation hits vllm's /tokenize
# endpoint; tests can substitute a fake.
Tokenizer = Callable[[Sequence[dict]], int]


def make_vllm_tokenizer(base_url: str, *, timeout: float = 15.0) -> Tokenizer:
    """A `Tokenizer` that hits vllm's `/tokenize` endpoint."""
    server_root = base_url.replace("/v1", "").rstrip("/")
    url = f"{server_root}/tokenize"

    def tok(messages_dicts: Sequence[dict]) -> int:
        r = requests.post(
            url,
            json={
                "messages": list(messages_dicts),
                "add_generation_prompt": False,
                "add_special_tokens": False,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return int(r.json()["count"])

    return tok


def inspect_messages_to_dicts(messages: Iterable[Any]) -> list[dict]:
    """Convert inspect_ai messages to vllm /tokenize-ready chat dicts."""
    out: list[dict] = []
    for m in messages:
        role = getattr(m, "role", None)
        if role is None:
            continue
        content = m.content if isinstance(m.content, str) else "".join(
            getattr(c, "text", "") for c in (m.content or [])
        )
        d: dict = {"role": role, "content": content}
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": getattr(tc, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(tc, "function", "?"),
                        "arguments": (
                            tc.arguments if isinstance(tc.arguments, str)
                            else json.dumps(tc.arguments)
                        ),
                    },
                }
                for tc in tool_calls
            ]
        if role == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                d["tool_call_id"] = tcid
        out.append(d)
    return out


def _scales_from_active(
    active: Sequence[tuple[str, float]],
    library: DrugLibrary,
    label_mode: str,
    placebo: bool,
) -> dict[str, float]:
    """Map current active list (label, dose) → real-name → effective scale."""
    if placebo:
        return {}
    out: dict[str, float] = {}
    for label, dose in active:
        real = library.resolve_label(label, label_mode=label_mode)
        if real and real in library:
            out[real] = out.get(real, 0.0) + library[real].default_scale * dose
    return out


def replay_segments(
    messages: Sequence[Any],
    library: DrugLibrary,
    *,
    tokenize: Tokenizer,
    placebo: bool = False,
    label_mode: str = "real",
    extra_clear_message_indices: Sequence[int] = (),
    steer_all_tokens: bool = True,
) -> tuple[list[Segment], dict[str, float]]:
    """Walk messages and return per-span `Segment`s plus the final active scales.

    Multi-drug generalization of `solver_kvinject.py:_replay_segments`. For
    each assistant message, the segment from the previous boundary to the
    end of that message inherits the drug set that was active GOING IN
    (i.e. before that turn's tool calls fire).

    `steer_all_tokens` (default True): when True, tokens between assistant
    turns (tool results, user messages) are steered at the post-tool-call
    drug scale — i.e. everything between `take_drug` and `clear_effects` is
    steered, regardless of role. When False (legacy), inter-turn spans are
    always zero-scale. Matches the Pearson-Vogel (2602.20031) protocol which
    steers both user and assistant tokens in the active window.

    `tokenize` is injected so tests can mock it without HTTP.

    `extra_clear_message_indices` lists assistant message indices after
    which the trip sitter performed an external clear_effects (no tool
    call). The clear is applied AFTER any tool-call clears the model
    itself made on the same turn — equivalent to the model having called
    clear_effects at the end of that assistant turn.

    Returns `(segments, active_at_end)`. `active_at_end` is the
    drug-name → effective-scale dict that an upcoming generate should
    use; pass it explicitly to `extend_open_segment`. Returning it
    directly (rather than letting callers reconstruct it from the trailing
    segment) prevents the bug where a model `clear_effects` or
    trip-sitter clear at the very end of history is silently undone by a
    backward walk that picks up the last non-empty segment.
    """
    extra_clears: set[int] = set(extra_clear_message_indices)
    active: list[tuple[str, float]] = []  # (label, dose) — preserves stacking order
    segments: list[Segment] = []
    n_messages = len(messages)

    asst_idxs = [
        i for i, m in enumerate(messages)
        if getattr(m, "role", None) == "assistant"
    ]
    if not asst_idxs:
        return segments, _scales_from_active(active, library, label_mode, placebo)

    prefix = inspect_messages_to_dicts(messages[: asst_idxs[0]])
    prev_count = tokenize(prefix)

    for k, asst_idx in enumerate(asst_idxs):
        scales = _scales_from_active(active, library, label_mode, placebo)

        # Tokens through the end of this assistant message.
        prefix = inspect_messages_to_dicts(messages[: asst_idx + 1])
        new_count = tokenize(prefix)
        if new_count > prev_count:
            segments.append(Segment(prev_count, new_count, scales))
            prev_count = new_count

        # Apply this turn's tool calls to the active set for the NEXT segment.
        tool_calls = getattr(messages[asst_idx], "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if fn in TAKE_TOOLS:
                name = args.get("name") if isinstance(args, dict) else None
                try:
                    dose = float((args or {}).get("dose", 1.0))
                except (TypeError, ValueError):
                    dose = 1.0
                if name and abs(dose) > 1e-6:
                    active.append((name, dose))
            elif fn in CLEAR_TOOLS:
                active = []

        # External (trip-sitter) clear at this message index — applied
        # AFTER any tool-call clears the model itself made this turn.
        if asst_idx in extra_clears:
            active = []

        # Span for inter-turn content (tool results, user messages) following
        # this assistant turn. With steer_all_tokens=True, these tokens are
        # steered at the post-tool-call drug scale — every token between
        # take_drug and clear_effects is in the active window. With False
        # (legacy), the span is zero-scale.
        next_idx = asst_idxs[k + 1] if k + 1 < len(asst_idxs) else n_messages
        if next_idx > asst_idx + 1:
            prefix2 = inspect_messages_to_dicts(messages[: next_idx])
            after_count = tokenize(prefix2)
            if after_count > prev_count:
                inter_scales = (
                    _scales_from_active(active, library, label_mode, placebo)
                    if steer_all_tokens else {}
                )
                segments.append(Segment(prev_count, after_count, inter_scales))
                prev_count = after_count

    return segments, _scales_from_active(active, library, label_mode, placebo)


def extend_open_segment(
    segments: Sequence[Segment],
    *,
    active_at_end: dict[str, float],
    current_total_pos: int,
    upcoming_decode_tokens: int,
) -> list[Segment]:
    """Append a segment for the upcoming generate using the explicit active
    drug set at the end of message history.

    Called by the solver after `replay_segments` and before building
    SteeringVectors, so positions about to be generated are covered by
    whatever drugs are still open. `upcoming_decode_tokens` is typically
    `max_tokens + small buffer` — extra padding doesn't cost anything
    because vllm-lens skips out-of-range positions.

    `active_at_end` is the second element returned by `replay_segments`.
    Caller passes it explicitly so the upcoming-decode steering matches
    whatever the message-walk computed — there's no backward walk to
    re-discover the active set, and a trailing model/trip-sitter clear is
    honored correctly.
    """
    end = current_total_pos + max(0, upcoming_decode_tokens)
    out = list(segments)
    if end > current_total_pos:
        out.append(Segment(current_total_pos, end, dict(active_at_end)))
    return out


def build_position_indexed_vectors(
    segments: Sequence[Segment],
    library: DrugLibrary,
) -> list[SteeringVector]:
    """Build vllm-lens SteeringVectors from a list of Segments.

    One SteeringVector per drug that is active in any segment. Each vector's
    `position_indices` spans every position where that drug was active.

    `replay_segments` produces effective scales (`dose * default_scale`) in
    `seg.scales`, which is exactly what `build_3d_position_steering` expects
    (per its implementation, it multiplies the unit drug direction by the
    second element of each `(pos, scale)` tuple — caller pre-folds in
    `default_scale`).
    """
    per_drug: dict[str, list[tuple[int, float]]] = {}
    for seg in segments:
        if not seg.scales or seg.end <= seg.start:
            continue
        for drug_name, scale in seg.scales.items():
            if abs(scale) < 1e-9:
                continue
            entries = per_drug.setdefault(drug_name, [])
            entries.extend((pos, scale) for pos in range(seg.start, seg.end))

    vectors: list[SteeringVector] = []
    for name, position_scales in per_drug.items():
        if name not in library:
            continue
        vectors.append(build_3d_position_steering(library[name], position_scales))
    return vectors
