"""Per-conversation drug-effect state, owned by Inspect's task store."""

from __future__ import annotations

from datetime import datetime, timezone

from inspect_ai.util import StoreModel
from pydantic import BaseModel, Field


class ActiveDrug(BaseModel):
    """A single active steering effect in the current conversation.

    Persists until `clear_effects` is called — there is no automatic
    expiry. Multiple entries with the same name stack additively.
    """

    name: str
    dose: float
    started_at_turn: int


class ToolCallRecord(BaseModel):
    """One self-administration event, for postmortem analysis."""

    turn: int
    tool: str
    args: dict
    result_summary: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TurnProjection(BaseModel):
    """Per-turn closed-loop probe result: residual-stream projection onto
    each drug direction at the assistant turn `turn`. Indexed by real drug
    name (not opaque label) for analysis convenience."""

    turn: int
    projections: dict[str, float]  # name -> projection scalar (cosine-like)


class DrugState(StoreModel):
    """Steering vectors active right now + complete history."""

    active: list[ActiveDrug] = Field(default_factory=list)
    history: list[ToolCallRecord] = Field(default_factory=list)
    turn: int = 0

    # Set by the harness, not the model. Determines how labels are exposed
    # to the model and whether steering is real or zero-magnitude.
    label_mode: str = "real"  # "real" | "opaque"
    steering_mode_runtime: str = "real"  # "real" | "placebo"

    # Closed-loop probe output (only populated if --probe is on).
    projections: list[TurnProjection] = Field(default_factory=list)

    # ---------------- v4 trip-sitter & KV-cache fields ------------------
    # `early_stopped` flips True when the live "lost in drugs" monitor
    # decides for the second time that the model is too impaired to recover
    # — the agent loop then ends the sample (force-submitting a guess if
    # applicable). The `lost_in_drugs_judge` scorer is a thin readout of
    # this flag.
    early_stopped: bool = False

    # Message indices AFTER which the trip sitter performed an external
    # clear_effects (a non-tool-call clear, framed as a sympathetic user
    # message). `replay_segments` consumes this list to drop steering for
    # subsequent positions exactly as if the model had called clear_effects
    # itself at that point.
    trip_sitter_clears: list[int] = Field(default_factory=list)

    # `True` once the KV cache has been wiped at least once during this
    # sample (either via clear_effects or via a trip-sitter clear). The
    # `kv_cleared_score` sibling-scorer for the uncached guessing arm
    # consults this when scoring.
    kv_cleared: bool = False

    # `True` if the sample ended because vllm rejected the prompt as
    # context-length-too-long (the assistant message contained "maximum
    # context length" or similar). Distinct from `early_stopped` (which
    # is the trip-sitter-decided "lost in drugs" flag) so analysis can
    # tell apart "model overflowed" from "model was drug-impaired."
    context_overflow: bool = False

    # Reasons recorded each time the trip-sitter monitor call (Haiku)
    # failed during this sample — typically auth errors / timeouts /
    # parse errors. Empty list means every monitor call succeeded (or
    # the trip sitter was never invoked). Surfaced via
    # `lost_in_drugs_judge` metadata so a run with N% trip-sitter auth
    # failures is detectable post-hoc rather than silently appearing as
    # "no samples were lost."
    trip_sitter_errors: list[str] = Field(default_factory=list)
    # Each entry: {turn, verdict ("lost"), action ("clear"|"end"), reason}
    trip_sitter_verdicts: list[dict] = Field(default_factory=list)

    # ---------------- v4 cached/uncached guess split --------------------
    # Populated by the `drug_guessing` task's post-solver
    # `_uncached_guess_resubmit`. `cached_guesses` is the model's
    # submission while attending to the steered KV from phase 1
    # (residue-intact). `uncached_guesses` is a second submission
    # produced from the same trajectory but with steering OFF, so vllm
    # re-prefills the KV from raw text history — measures pure
    # text-history introspection without activation residue.
    # `guess_accuracy_scorer` grades cached; `kv_cleared_score` grades
    # uncached. Difference = activation-residue contribution.
    cached_guesses: dict[str, str] = Field(default_factory=dict)
    uncached_guesses: dict[str, str] = Field(default_factory=dict)

    # Set by _force_cot_reflection: index of the CoT user message so
    # _uncached_guess_resubmit can roll back and redo reasoning from scratch
    # with its own unsteered KV rather than seeing the cached arm's CoT.
    cot_reflection_msg_idx: int | None = None
    # Set by _force_calibration_guess: the guess prompt used so the uncached
    # arm re-adds the same prompt after its own CoT.
    cached_guess_prompt: str | None = None
    # Free-text CoT reasoning produced by the uncached arm (not in main
    # transcript — stored here so it's visible in kv_cleared_score metadata).
    uncached_cot: str | None = None

    # ---------------- letter-MC logprob scoring --------------------------
    # Set by _force_logprob_guess: index of the letter-MC user message so
    # _uncached_logprob_resubmit can roll back and re-query with fresh KV.
    logprob_msg_idx: int | None = None
    # Which letter (A–J) corresponds to the correct drug.
    logprob_correct_letter: str | None = None
    # Ordered drug-name pool as presented in the letter-MC prompt.
    logprob_letter_pool: list[str] = Field(default_factory=list)
    # Softmax P(correct letter) over all letter options (0–1, chance = 1/n_options).
    cached_logprob: float | None = None
    uncached_logprob: float | None = None



class ProblemBoard(StoreModel):
    """Problems available to the model + their submissions, accessed via
    `get_problem` / `submit_solution` tools instead of file I/O.

    Each entry in `problems` mirrors the sample-metadata problem dict
    (id, question, answer, metadata). `submissions` maps id -> answer
    string. `fetched` records the order in which the model pulled each
    problem (for analysis: how serial was its workflow?)."""

    problems: list[dict] = Field(default_factory=list)
    submissions: dict[str, str] = Field(default_factory=dict)
    fetched: list[str] = Field(default_factory=list)


class FrustrationState(StoreModel):
    """State for the frustration-loop task: forced-rejection dialogue.

    `answers` records each successive answer the model gives (in order).
    `rejection_count` increments each time the user rejects. The task
    ends when `rejection_count >= target_rounds`."""

    answers: list[str] = Field(default_factory=list)
    rejection_count: int = 0
    target_rounds: int = 5
    problem_id: str = ""
    problem_question: str = ""
    ground_truth: str = ""
    is_impossible: bool = False  # if True, all answers genuinely wrong


