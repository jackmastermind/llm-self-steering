"""Smoke tests for src/hackday/agent/kv_steering.py.

Pure-Python — no GPU, no HTTP. We mock the tokenizer and construct a
minimal in-memory DrugLibrary. Run with `pytest tests/test_kv_steering.py`
or directly: `python tests/test_kv_steering.py`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
# vllm-lens submodule isn't installed (uv sync requires an ARM C compiler not
# available on this host); use the source tree directly for tests.
sys.path.insert(0, str(REPO_ROOT / "vllm-lens"))

from hackday.agent.kv_steering import (  # noqa: E402
    Segment,
    build_position_indexed_vectors,
    extend_open_segment,
    inspect_messages_to_dicts,
    replay_segments,
)
from hackday.drugs.library import Drug, DrugLibrary  # noqa: E402


# --- fixtures ----------------------------------------------------------------

HIDDEN_DIM = 8
LAYER = 24


def _make_drug(name: str, default_scale: float = 2.0) -> Drug:
    """Synthetic Drug with a one-hot direction so we can verify activations
    later if we want."""
    vec = torch.zeros(HIDDEN_DIM, dtype=torch.float32)
    vec[hash(name) % HIDDEN_DIM] = 1.0
    return Drug(
        name=name,
        vectors_by_layer={LAYER: vec},
        default_scale=default_scale,
        description=f"test drug {name}",
        apply_layers=[LAYER],
    )


def _make_library(*names: str, default_scale: float = 2.0) -> DrugLibrary:
    return DrugLibrary({n: _make_drug(n, default_scale) for n in names})


@dataclass
class _Msg:
    """Minimal stand-in for an inspect_ai message. role + content + optional
    tool_calls."""

    role: str
    content: str = ""
    tool_calls: list | None = None
    tool_call_id: str | None = None


def _tc(function: str, **args):
    return SimpleNamespace(id="tc1", function=function, arguments=args)


# --- inspect_messages_to_dicts -----------------------------------------------


def test_inspect_messages_to_dicts_basic():
    msgs = [
        _Msg("system", "hi"),
        _Msg("user", "hello"),
        _Msg("assistant", "ok", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),
        _Msg("tool", "done", tool_call_id="tc1"),
    ]
    out = inspect_messages_to_dicts(msgs)
    assert len(out) == 4
    assert out[0] == {"role": "system", "content": "hi"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "take_drug"
    # arguments must be a JSON string per OpenAI/vllm schema
    assert isinstance(out[2]["tool_calls"][0]["function"]["arguments"], str)
    assert out[3]["tool_call_id"] == "tc1"


# --- replay_segments ---------------------------------------------------------


def _fake_tokenizer(message_token_counts: dict[int, int]):
    """Returns a tokenizer that maps len(messages_dicts) → cumulative token
    count via the provided mapping. Lookup falls back to len * 10."""

    def tok(messages_dicts):
        n = len(messages_dicts)
        return message_token_counts.get(n, n * 10)

    return tok


def test_replay_segments_no_assistants():
    lib = _make_library("creative")
    segs, active_at_end = replay_segments(
        [_Msg("system", "hi"), _Msg("user", "hello")],
        lib,
        tokenize=_fake_tokenizer({}),
    )
    assert segs == []
    assert active_at_end == {}


def test_replay_segments_take_then_clear():
    """Walk: system → user → assistant(take_drug) → tool → assistant(write)
    → tool(none) → assistant(clear_effects) → assistant(write again).
    Verify segment scales reflect what was active during each assistant turn.
    """
    lib = _make_library("creative", default_scale=2.0)  # effective for dose=1 → 2.0
    msgs = [
        _Msg("system", "sys"),                                              # 0
        _Msg("user", "hi"),                                                 # 1
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),  # 2
        _Msg("tool", "took it", tool_call_id="tc1"),                       # 3
        _Msg("assistant", "feeling steered"),                              # 4
        _Msg("assistant", "", tool_calls=[_tc("clear_effects")]),          # 5
        _Msg("assistant", "sober now"),                                    # 6
    ]
    # Cumulative token counts at each prefix length (len of dicts list):
    # 2 → 100 (before first asst)
    # 3 → 110 (asst#1, the take_drug call — generated WITHOUT steering)
    # 4 → 115 (after tool result)
    # 5 → 200 (asst#2, generated WITH creative active — scale 2.0)
    # 6 → 280 (asst#3, the clear_effects call — still under creative
    #          because clear_effects fires AFTER generation)
    # 7 → 350 (asst#4, sober)
    counts = {2: 100, 3: 110, 4: 115, 5: 200, 6: 280, 7: 350}
    segs, active_at_end = replay_segments(msgs, lib, tokenize=_fake_tokenizer(counts))

    # steer_all_tokens=True (default): inter-turn span after take_drug is steered.
    assert [s.start for s in segs] == [100, 110, 115, 200, 280]
    assert [s.end for s in segs] == [110, 115, 200, 280, 350]
    assert segs[0].scales == {}                       # asst#1: pre-take_drug
    assert segs[1].scales == {"creative": 2.0}        # tool-result span: drug active
    assert segs[2].scales == {"creative": 2.0}        # asst#2: under creative
    assert segs[3].scales == {"creative": 2.0}        # asst#3: clear hadn't fired yet
    assert segs[4].scales == {}                       # asst#4: sober
    assert active_at_end == {}

    # steer_all_tokens=False (legacy): inter-turn span is always zero-scale.
    segs_legacy, _ = replay_segments(msgs, lib, tokenize=_fake_tokenizer(counts),
                                     steer_all_tokens=False)
    assert segs_legacy[1].scales == {}                # tool-result span: zero (legacy)


def test_replay_segments_multi_drug_stacking():
    lib = _make_library("creative", "focused", default_scale=2.0)
    msgs = [
        _Msg("system", "sys"),
        _Msg("user", "go"),
        _Msg("assistant", "", tool_calls=[
            _tc("take_drug", name="creative", dose=1.0),
            _tc("take_drug", name="focused", dose=0.5),
        ]),
        _Msg("assistant", "writing under both"),
    ]
    counts = {2: 50, 3: 60, 4: 100}
    segs, active_at_end = replay_segments(msgs, lib, tokenize=_fake_tokenizer(counts))
    # asst#1: nothing active yet (take_drug is not effective until the
    # tool call has been processed)
    assert segs[0].scales == {}
    # asst#2: both active — creative @ 1.0 → 2.0, focused @ 0.5 → 1.0
    assert segs[1].scales == {"creative": 2.0, "focused": 1.0}
    assert active_at_end == {"creative": 2.0, "focused": 1.0}


def test_replay_segments_same_drug_stacks_additively():
    lib = _make_library("creative", default_scale=2.0)
    msgs = [
        _Msg("user", "go"),
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=0.5)]),
        _Msg("assistant", "writing under stacked creative"),
    ]
    counts = {1: 20, 2: 30, 3: 40, 4: 80}
    segs, active_at_end = replay_segments(msgs, lib, tokenize=_fake_tokenizer(counts))
    # asst#3 sees creative @ 1.0 + creative @ 0.5 → effective 1.5 * 2.0 = 3.0
    assert segs[-1].scales == {"creative": 3.0}
    assert active_at_end == {"creative": 3.0}


def test_replay_segments_placebo_zeroes_scales():
    lib = _make_library("creative")
    msgs = [
        _Msg("user", "go"),
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),
        _Msg("assistant", "would be steered, but placebo"),
    ]
    counts = {1: 20, 2: 30, 3: 60}
    segs, active_at_end = replay_segments(
        msgs, lib, tokenize=_fake_tokenizer(counts), placebo=True
    )
    assert all(s.scales == {} for s in segs)
    assert active_at_end == {}


def test_replay_segments_opaque_label_resolution():
    # opaque mode: model calls take_drug("d1", ...) and we must resolve
    # to the underlying real drug name (creative).
    lib = _make_library("creative", "focused", default_scale=2.0)
    msgs = [
        _Msg("user", "go"),
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="d1", dose=1.0)]),
        _Msg("assistant", "under d1=creative"),
    ]
    counts = {1: 10, 2: 20, 3: 50}
    segs, active_at_end = replay_segments(
        msgs, lib, tokenize=_fake_tokenizer(counts), label_mode="opaque"
    )
    # d1 → creative
    assert segs[-1].scales == {"creative": 2.0}
    assert active_at_end == {"creative": 2.0}


# --- trip-sitter / clear regression tests -----------------------------------


def test_replay_segments_trip_sitter_clear_wipes_active():
    """Regression: trip-sitter clear at an assistant index drops active drugs
    for ALL subsequent generation. Without this, the sympathetic 'I cleared
    your drugs' user message lies — steering keeps applying to new tokens."""
    lib = _make_library("creative", default_scale=2.0)
    msgs = [
        _Msg("user", "go"),                                                    # 0
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),  # 1
        _Msg("tool", "took", tool_call_id="tc1"),                              # 2
        _Msg("assistant", "stuck looping under creative"),                     # 3 ← trip sitter fires here
        _Msg("user", "(sympathetic clear msg)"),                               # 4
        _Msg("assistant", "ok, sober now"),                                    # 5
    ]
    counts = {1: 10, 2: 20, 3: 25, 4: 60, 5: 70, 6: 100}
    segs, active_at_end = replay_segments(
        msgs,
        lib,
        tokenize=_fake_tokenizer(counts),
        extra_clear_message_indices=[3],  # trip sitter recorded asst#3 as clear
    )
    # Asst#3's own segment was generated UNDER steering (clear fires AFTER).
    asst3_seg = next(s for s in segs if s.start == 25 and s.end == 60)
    assert asst3_seg.scales == {"creative": 2.0}
    # Asst#5 (post-clear) is sober.
    asst5_seg = next(s for s in segs if s.start == 70 and s.end == 100)
    assert asst5_seg.scales == {}
    # And critically: active_at_end must reflect the trip-sitter clear, so
    # the upcoming decode in `extend_open_segment` is unsteered.
    assert active_at_end == {}


def test_replay_segments_clear_at_end_yields_empty_active_at_end():
    """Regression: model calls clear_effects on the FINAL assistant turn (no
    further take_drug, no further assistant). active_at_end must be empty
    even though the last segment with non-empty scales was creative."""
    lib = _make_library("creative", default_scale=2.0)
    msgs = [
        _Msg("user", "go"),                                                    # 0
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),  # 1
        _Msg("assistant", "writing under creative"),                           # 2
        _Msg("assistant", "", tool_calls=[_tc("clear_effects")]),              # 3
    ]
    counts = {1: 10, 2: 20, 3: 50, 4: 60}
    segs, active_at_end = replay_segments(msgs, lib, tokenize=_fake_tokenizer(counts))
    # Asst#2 (writing) was steered; asst#3 (the clear) was generated under
    # creative because clear fires AFTER generation.
    assert any(s.scales == {"creative": 2.0} for s in segs)
    # But by end of history we're sober.
    assert active_at_end == {}


def test_replay_segments_take_after_trip_sitter_clear_opens_fresh_range():
    """After a trip-sitter clear, a subsequent take_drug should open a brand
    new range — not stack on top of the cleared-out drugs."""
    lib = _make_library("creative", "focused", default_scale=2.0)
    msgs = [
        _Msg("user", "go"),                                                    # 0
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="creative", dose=1.0)]),  # 1
        _Msg("assistant", "writing under creative"),                           # 2 ← trip sitter
        _Msg("user", "(sympathetic clear)"),                                   # 3
        _Msg("assistant", "", tool_calls=[_tc("take_drug", name="focused", dose=1.0)]),   # 4
        _Msg("assistant", "writing under focused only"),                       # 5
    ]
    counts = {1: 10, 2: 20, 3: 50, 4: 60, 5: 70, 6: 100}
    segs, active_at_end = replay_segments(
        msgs,
        lib,
        tokenize=_fake_tokenizer(counts),
        extra_clear_message_indices=[2],  # trip sitter recorded asst#2 as clear
    )
    # Asst#5 sees focused only — creative was wiped by the trip-sitter clear.
    asst5_seg = next(s for s in segs if s.start == 70 and s.end == 100)
    assert asst5_seg.scales == {"focused": 2.0}
    assert active_at_end == {"focused": 2.0}


# --- extend_open_segment -----------------------------------------------------


def test_extend_open_segment_uses_active_at_end():
    # extend_open_segment now takes the active set explicitly, so the
    # caller (replay_segments) decides what's active for the upcoming
    # decode. No backward walk over segments.
    segs = [
        Segment(0, 100, {"creative": 2.0}),
        Segment(100, 110, {}),  # tool-result span
    ]
    out = extend_open_segment(
        segs,
        active_at_end={"creative": 2.0},
        current_total_pos=110,
        upcoming_decode_tokens=512,
    )
    assert len(out) == 3
    assert out[2].start == 110 and out[2].end == 622
    assert out[2].scales == {"creative": 2.0}


def test_extend_open_segment_empty_when_no_drugs_active():
    segs = [Segment(0, 50, {})]
    out = extend_open_segment(
        segs,
        active_at_end={},
        current_total_pos=50,
        upcoming_decode_tokens=200,
    )
    assert out[1].scales == {}


def test_extend_open_segment_zero_decode_skips():
    segs = [Segment(0, 50, {"creative": 2.0})]
    out = extend_open_segment(
        segs,
        active_at_end={"creative": 2.0},
        current_total_pos=50,
        upcoming_decode_tokens=0,
    )
    assert out == list(segs)


def test_extend_open_segment_after_clear_does_not_reinherit():
    # Regression: after a clear_effects (or trip-sitter clear), the
    # PRIOR steered segment is still in `segments` (history is immutable),
    # but the upcoming decode should be sober. extend_open_segment must
    # use the explicit active_at_end={} rather than walking backward.
    segs = [
        Segment(0, 100, {"creative": 2.0}),  # historical, model wrote under steering
        Segment(100, 110, {}),                # tool-result span for the clear
    ]
    out = extend_open_segment(
        segs,
        active_at_end={},  # what replay_segments computed post-clear
        current_total_pos=110,
        upcoming_decode_tokens=512,
    )
    # Upcoming-decode segment must be unsteered.
    assert out[-1].scales == {}


# --- build_position_indexed_vectors ------------------------------------------


def test_build_position_indexed_vectors_per_drug():
    lib = _make_library("creative", "focused", default_scale=2.0)
    segs = [
        Segment(0, 10, {}),                                  # unsteered
        Segment(10, 20, {"creative": 2.0}),
        Segment(20, 25, {"creative": 2.0, "focused": 1.0}),  # both active
        Segment(25, 30, {"focused": 1.0}),
    ]
    vecs = build_position_indexed_vectors(segs, lib)
    by_name = {}
    # vllm-lens SteeringVector has layer_indices but not the drug name,
    # so we identify via its activation direction instead.
    creative_dir = lib["creative"].vector
    focused_dir = lib["focused"].vector
    for v in vecs:
        # 3D activations: (n_layers=1, n_pos, hidden_dim). Take any non-zero
        # row and check which direction it aligns with.
        row = v.activations[0, 0]
        if torch.allclose(row / row.norm().clamp_min(1e-9), creative_dir):
            by_name["creative"] = v
        elif torch.allclose(row / row.norm().clamp_min(1e-9), focused_dir):
            by_name["focused"] = v
    assert set(by_name) == {"creative", "focused"}

    cv = by_name["creative"]
    fv = by_name["focused"]
    # creative: positions 10-19 + 20-24 = 15 positions, all scale 2.0
    assert cv.position_indices == list(range(10, 25))
    assert cv.activations.shape == (1, 15, HIDDEN_DIM)
    # row magnitude should equal scale 2.0 (one-hot direction × 2.0)
    assert torch.allclose(cv.activations[0, 0].norm(), torch.tensor(2.0))

    # focused: positions 20-24 + 25-29 = 10 positions, scale 1.0
    assert fv.position_indices == list(range(20, 30))
    assert fv.activations.shape == (1, 10, HIDDEN_DIM)
    assert torch.allclose(fv.activations[0, 0].norm(), torch.tensor(1.0))


def test_build_position_indexed_vectors_empty_when_unsteered():
    lib = _make_library("creative")
    segs = [Segment(0, 100, {})]
    assert build_position_indexed_vectors(segs, lib) == []


def test_build_position_indexed_vectors_skips_unknown_drug():
    lib = _make_library("creative")
    segs = [Segment(0, 10, {"creative": 2.0, "ghost": 1.0})]
    vecs = build_position_indexed_vectors(segs, lib)
    assert len(vecs) == 1  # only creative, ghost is dropped silently


# --- main entry point --------------------------------------------------------

if __name__ == "__main__":
    import traceback
    failures = 0
    for name in sorted(globals()):
        if not name.startswith("test_"):
            continue
        fn = globals()[name]
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
            print()
        else:
            print(f"PASS {name}")
    print()
    print(f"{failures} failures" if failures else "all passed")
    sys.exit(1 if failures else 0)
