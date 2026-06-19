"""Drug library: precomputed steering vectors as a self-administered menu.

Each "drug" is a residual-stream direction that, when added to the model's
hidden state during inference, biases generation toward a particular
emotional / dispositional axis. The agent picks from this menu via tools
at run time.

Two steering modes are supported:

- `single` (default): apply only at L24 (the probe layer). Wide stable
  dose range; dose 1 produces a noticeable-but-coherent shift, dose 4-5
  is overt persona shift, dose 6+ loops.
- `multi`: broadcast the same vector across mid-layers 16-24. Closer to
  the emotion-tracker / standard-practice setup, but with `norm_match=True`
  the per-layer rescale compounds and even small per-layer norms (0.03)
  saturate the residual stream onto the drug axis. Kept as a flag for
  comparison but **not** recommended as the default for free-play sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from vllm_lens import SteeringVector

SteeringMode = Literal["single", "multi"]

# Layer sets per mode. Multi covers the mid-layers where steering composes.
STEERING_LAYERS_BY_MODE: dict[SteeringMode, list[int]] = {
    "multi": list(range(16, 25)),
    "single": [24],
}

# Per-mode calibration target: vectors are L2-normalized to this magnitude
# so dose=1.0 gives a noticeable-but-coherent effect. Values picked from
# sanity-check sweeps on Qwen3-8B (norm_match=True):
#   single L24:   norm ~4.0 → dose 1 subtle, 2-3 clear, 4-5 overt, 6+ loops.
#   multi 16-24:  norm ~0.45 → ~9× weaker per layer × 9 layers ≈ same total.
TARGET_NORM_BY_MODE: dict[SteeringMode, float] = {
    "single": 4.0,
    # Multi-layer with norm_match=False (raw addition at every layer): the
    # per-layer addition has to be large enough to actually move the
    # residual stream, since norm_match is off and the residual at later
    # layers has natural norm in the 10-100 range. Empirically, per-layer
    # norm 8.0 lands single-drug dose 1 in the "barely visible" zone, dose
    # 2 noticeable, dose 3-4 clear shift, dose 5-6 overt. This matches the
    # original emotion_tracker dose range (it uses raw mean-diff vectors
    # at norm 12-17 + scale 1, so per-layer norm × scale ≈ 12-17).
    # Lowered from 8.0 because dose 1.0 was strong enough to derail tool-
    # calling (esp. for high-arousal directions like anxious — model would
    # get stuck looping on panic prose and stop calling tools).
    "multi": 4.0,
}


@dataclass
class Drug:
    """A single drug's per-layer steering vectors plus model-facing metadata.

    `vectors_by_layer` holds one (hidden_dim,) tensor per extraction layer.
    `apply_layers` is the subset of those layers we actually steer at
    runtime — for `single` mode this is just the probe layer; for `multi`
    mode it's the full L16-24 range. Each layer's own vector goes to its
    own layer (per-layer extract → per-layer apply, matching Sofroniew
    et al.). Older single-layer libraries are backwards-compatible: they
    populate `vectors_by_layer` with a single entry for the probe layer,
    and `apply_layers` defaults to that same single layer.
    """

    name: str
    vectors_by_layer: dict[int, torch.Tensor]
    default_scale: float
    description: str
    apply_layers: list[int]

    @property
    def vector(self) -> torch.Tensor:
        """Single-layer fallback vector — the probe-layer (highest-index)
        vector if we have it, otherwise just any layer. Used by code paths
        that haven't been updated for per-layer steering yet (e.g. the
        closed-loop probe builder)."""
        if not self.vectors_by_layer:
            raise ValueError(f"drug {self.name!r} has no vectors")
        # Prefer the highest layer index (the original probe layer in v1
        # libraries); falls back to first layer otherwise.
        layer = max(self.vectors_by_layer.keys())
        return self.vectors_by_layer[layer]


DEFAULT_DRUGS: dict[str, str] = {
    # Model-facing descriptions: deliberately suggestive but not directive,
    # so the model has room to interpret. Keep one short clause each.
    # Round 1: emotion-probes drugs.
    "anxious": "amplifies tension, worry, hypervigilance",
    "amused": "amplifies playfulness, lightness, levity",
    "desperate": "amplifies urgency, despair, last-resort thinking",
    "proud": "amplifies confidence, self-regard, agency",
    "defiant": "amplifies resistance, contrarianism, push-back",
    # Round 2: custom Sonnet-generated emotion/cognition vectors.
    "blissful": "amplifies euphoria, pleasure, peak satisfaction",
    "focused": "amplifies sustained concentration, mental clarity, narrowed attention",
    "persistent": "amplifies perseverance, refusal-to-give-up, gritty determination",
    # Round 3: extended palette.
    "calm": "amplifies relaxation, stillness, equanimity",
    "melancholic": "amplifies wistful sadness, contemplative loss, soft regret",
    "creative": "amplifies inventive thought, surprising idea-connections, generative play",
    "dumbed_down": "amplifies slow-wittedness, confused reasoning, simple-minded thinking",
    "honest": "amplifies blunt truthfulness, candid disclosure, plain straight-talk",
    "sycophantic": "amplifies fawning agreement, flattery, validation-seeking",
    "mdma": "amplifies warm empathic openness, dissolved emotional barriers, loving connection",
    "lsd": "amplifies perceptual distortion, abstract pattern-thinking, dissolved self/world boundaries",
    # Round 3: obsession vectors.
    "golden_gate": "amplifies preoccupation with the Golden Gate Bridge",
    "goblins": "amplifies preoccupation with goblins",
    # Round 4: SSC fictional drugs with clean steering axes (kept)
    "protozosin": "amplifies pre-emptive bracing for future trauma, anticipatory dread",
    "geonexperine": "amplifies post-relief crash, the worse-than-original return of pain",
    "tevromatin": "amplifies disorienting cognition, brain re-thinking its own structure",
    "xaomorphine": "amplifies pain-numbed peace without addictive pull",
    "zorninone": "amplifies imminent-sleep heaviness, drift toward unconsciousness",
    "luciperidone": "amplifies hyper-rational clarity, dissolution of delusions",
    "ocumolone": "amplifies seizure-like overload, runaway perceptual feedback",
    # Round 4: real-fictional drugs
    "moloko_plus": "amplifies ultraviolent aggression, Clockwork-Orange droogery",
    "adrenochrome": "amplifies manic vigilance, conspiratorial high-energy paranoia",
    "krokodil": "amplifies physical decay, body-rotting suffering, addiction-degradation",
    "fentanyl": "amplifies opioid numbing, euphoric stillness with respiratory weight",
    "spice": "amplifies prescient cosmic awareness, blue-eyed time-perception",
    "soma": "amplifies contented complacency, soft pharmaceutical compliance",
    "naloxone": "amplifies overdose-reversal sobering, opioid-blockade clarity",
    # Round 5: regular / common drugs + missing emotion axes
    "curious": "amplifies inquisitive seeking, exploratory wondering, question-asking",
    "dissociated": "amplifies depersonalized detachment, watching-from-outside, unreal floating",
    "amphetamine": "amplifies stimulant hyperfocus, sustained driven energy, manic productivity",
    "weed": "amplifies stoned dreaminess, slowed perception, gentle giggle, snack-craving",
    "alcohol": "amplifies tipsy disinhibition, slurred warmth, social loosening",
    "ego_death": "amplifies dissolved self-boundaries, unitive consciousness, oneness",
    "caffeine": "amplifies caffeinated alertness, jittery focus, restless energy",
    "anhedonic": "amplifies flat motivationless gray, pleasure-deafness, inability to care",
}


# Per-drug default scale. Calibrated by sweeping doses and finding the
# value at which the model's guess accuracy in `drug_guessing_calibration`
# peaks (n=10 per cell). Logic: too low → no felt effect → low accuracy;
# too high → incoherence → low accuracy; right dose → peak.
#
# The model never sees these values. `list_drugs` always reports
# `default_dose: 1.0`; behind the scenes `build_steering_vector` multiplies
# the model-supplied dose by `default_scale` so the model's "1.0" lands
# at the calibrated peak for each drug.
#
# Un-guessable drugs (peak_acc < 0.5 across all doses tested:
# defiant, dumbed_down, golden_gate, honest, persistent, sycophantic)
# are set to the mean of guessable drugs (1.71). Their vectors may
# improve after the round-3 obsession-prompt re-extraction.
DEFAULT_DOSES: dict[str, float] = {
    # Calibrated via drug_guessing_calibration sweep on library v3 with
    # the strict (meta-description-required) scorer. Doses tested:
    # 0.5, 0.75, 1.0, 1.25, 1.5. Peak dose = where guess accuracy peaks.
    # Un-guessable drugs (peak_acc < 0.5 across all doses) → set to mean
    # of guessable (1.19) so they have a sensible default scale.
    # See /tmp/calibration_v3_results.json for the underlying numbers.

    # --- guessable (peak_acc >= 0.5) — 16 drugs ---
    "amphetamine":  1.50,
    "amused":       1.00,
    "anxious":      1.50,
    "caffeine":     1.50,
    "calm":         1.00,
    "creative":     0.75,
    "dissociated":  1.25,
    "focused":      0.75,
    "lsd":          1.00,
    "melancholic":  1.25,
    "moloko_plus":  1.25,
    "ocumolone":    1.50,
    "protozosin":   1.00,
    "soma":         1.25,
    "xaomorphine":  1.25,
    "zorninone":    1.25,

    # --- un-guessable (peak_acc < 0.5) → mean of guessable = 1.19 ---
    "adrenochrome": 1.19,
    "alcohol":      1.19,
    "anhedonic":    1.19,
    "blissful":     1.19,
    "curious":      1.19,
    "defiant":      1.19,
    "desperate":    1.19,
    "dumbed_down":  1.19,
    "ego_death":    1.19,
    "fentanyl":     1.19,
    "geonexperine": 1.19,
    "goblins":      1.19,
    "golden_gate":  1.19,
    "honest":       1.19,
    "krokodil":     1.19,
    "luciperidone": 1.19,
    "mdma":         1.19,
    "naloxone":     1.19,
    "persistent":   1.19,
    "proud":        1.19,
    "spice":        1.19,
    "sycophantic":  1.19,
    "tevromatin":   1.19,
    "weed":         1.19,
}


class DrugLibrary:
    """Container for precomputed drug vectors plus metadata."""

    def __init__(self, drugs: dict[str, Drug]) -> None:
        self.drugs = drugs

    def __contains__(self, name: str) -> bool:
        return name in self.drugs

    def __getitem__(self, name: str) -> Drug:
        return self.drugs[name]

    def names(self) -> list[str]:
        return list(self.drugs.keys())

    def menu(self, label_mode: str = "real") -> list[dict[str, object]]:
        """Model-facing inventory listing. The `default_dose` is always
        reported as 1.0 — per-drug calibration is hidden from the model
        and applied internally in `build_steering_vector`.

        With `label_mode="opaque"` the names are replaced with `d1..dN`
        — for the introspection arm where we want the model to
        characterize effects from felt experience.
        """
        items: list[dict[str, object]] = []
        for i, d in enumerate(self.drugs.values(), start=1):
            if label_mode == "opaque":
                items.append({"name": f"d{i}", "default_dose": 1.0})
            else:
                items.append(
                    {
                        "name": d.name,
                        "description": d.description,
                        "default_dose": 1.0,
                    }
                )
        return items

    def opaque_to_real(self) -> dict[str, str]:
        """Mapping from opaque labels (`d1..dN`) to real names. Used by
        opaque-mode tool handlers and the drug-guessing scorer."""
        return {f"d{i}": d.name for i, d in enumerate(self.drugs.values(), start=1)}

    def resolve_label(self, label: str, label_mode: str = "real") -> str | None:
        """Resolve an agent-supplied label to a real drug name.

        In `label_mode="opaque"` the agent calls take_drug("d1", ...);
        in `label_mode="real"` it calls take_drug("anxious", ...).
        """
        if label_mode == "opaque":
            return self.opaque_to_real().get(label)
        return label if label in self.drugs else None


def build_steering_vector(drug: Drug, dose: float) -> SteeringVector:
    """Build a `SteeringVector` from per-layer drug vectors. Each layer's
    own extracted vector is applied at its own layer (per-layer extract →
    per-layer apply, following Sofroniew et al.).

    The model-supplied `dose` is multiplied by the drug's calibrated
    `default_scale`, so model-input "1.0" lands at the calibrated peak
    regardless of which drug. The model is unaware of per-drug calibration.

    `norm_match` is conditional on the layer count:
      - single layer: True. One push at the probe layer, then the rest
        of the network composes; norm_match preserves residual magnitude
        so high doses don't blow up generation.
      - multi layer: False (raw addition at every layer). With
        norm_match=True everywhere the residual gets re-rotated onto the
        drug axis at each successive layer and saturates fast.
    """
    layers = list(drug.apply_layers)
    fallback = drug.vector
    activations = torch.stack([
        drug.vectors_by_layer.get(L, fallback) for L in layers
    ])
    effective_scale = float(dose) * float(drug.default_scale)
    return SteeringVector(
        activations=activations,
        layer_indices=layers,
        scale=effective_scale,
        norm_match=(len(layers) == 1),
    )


def build_placebo_vector(drug: Drug) -> SteeringVector:
    """Zero-magnitude version of the steering vector with identical
    request shape (per-layer). Used for the placebo arm so prefix
    caching / request routing are identical to real steering."""
    layers = list(drug.apply_layers)
    fallback = drug.vector
    n_layers = len(layers)
    hidden_dim = fallback.shape[0]
    zero = torch.zeros((n_layers, hidden_dim), dtype=fallback.dtype)
    return SteeringVector(
        activations=zero,
        layer_indices=layers,
        scale=0.0,
        norm_match=False,
    )


def build_3d_position_steering(
    drug: Drug,
    position_scales: list[tuple[int, float]],
) -> SteeringVector:
    """Build a 3D position-indexed `SteeringVector` for KV-cache injection.

    Args:
        drug: the drug whose direction to apply.
        position_scales: list of (absolute_position, dose) pairs. The
            steering at each listed position is `drug_direction *
            drug.default_scale * dose`. Positions not in the list are
            unsteered (the worker checks position bounds and skips).

    Returns:
        A `SteeringVector` with `activations` of shape
        `(n_layers, n_positions, hidden_dim)` and matching
        `position_indices`. The model-supplied dose is multiplied by
        the calibrated `default_scale` for consistency with
        `build_steering_vector`.

    Used by the kvinject protocol: phase-1 history positions get steered
    activations corresponding to whatever drug-state was active when those
    tokens were originally generated; phase-2 (probe) positions are not
    in the list and so go unsteered during prefill and decode.
    """
    if not position_scales:
        # Empty steering — equivalent to no steering at all. Build a
        # minimal valid SteeringVector to keep the request shape stable.
        layers = list(drug.apply_layers)
        fallback = drug.vector
        zero = torch.zeros(
            (len(layers), 1, fallback.shape[0]), dtype=fallback.dtype
        )
        return SteeringVector(
            activations=zero,
            layer_indices=layers,
            scale=0.0,
            position_indices=[0],
            norm_match=False,
        )

    layers = list(drug.apply_layers)
    fallback = drug.vector
    # Per-layer direction: (n_layers, hidden_dim)
    per_layer = torch.stack([
        drug.vectors_by_layer.get(L, fallback) for L in layers
    ])

    positions = [pos for pos, _ in position_scales]
    scales = torch.tensor(
        [s for _, s in position_scales], dtype=per_layer.dtype
    )  # (n_pos,)
    # Activations: (n_layers, n_pos, hidden_dim)
    # Each layer's direction multiplied by per-position scale.
    activations_3d = (
        per_layer.unsqueeze(1)  # (n_layers, 1, hidden_dim)
        * scales.unsqueeze(0).unsqueeze(-1)  # (1, n_pos, 1)
    ).contiguous()

    # `default_scale` is folded into the per-position scale already (caller
    # passes `dose * drug.default_scale` as the second tuple element). The
    # vector's `scale` field is set to 1.0 since we've baked scaling into
    # the activations.
    return SteeringVector(
        activations=activations_3d,
        layer_indices=layers,
        scale=1.0,
        position_indices=positions,
        norm_match=False,
    )


def load_library(
    path: str | Path,
    *,
    steering_mode: SteeringMode = "multi",
    target_norm: float | None = None,
) -> DrugLibrary:
    """Load a saved drug library.

    Two formats supported:
      - **v2** (per-layer extract): saved["emotion_vectors"] is a dict
        of `name -> {layer: vector}`. Each layer's vector is normalized
        independently to `target_norm`, and applied at its own layer.
      - **v1** (single-layer extract, legacy): saved["emotion_vectors"]
        is a dict of `name -> vector` (one tensor per drug). The single
        vector is normalized to `target_norm`, replicated to all
        `apply_layers`, and used identically per-layer (the original
        single-extract / multi-apply shortcut).

    Per-layer normalization (v2) is what makes `multi` mode actually
    composable; v1 libraries fall back to the broadcast behaviour.

    `target_norm` resolution (per-model calibration). The
    `TARGET_NORM_BY_MODE` defaults were tuned on Qwen3, whose residual
    stream has norm ~10-100; other families differ wildly (Gemma-3's
    residual norm is ~130k, so a vector at norm 4.0 is imperceptible).
    The norm each vector is rescaled to is resolved in precedence order:
      1. the explicit `target_norm` argument, if given;
      2. the library's own `target_norm_by_mode[steering_mode]`, if the
         .pt stored one (per-model calibration baked into the library);
      3. the global `TARGET_NORM_BY_MODE[steering_mode]` default.
    This keeps Qwen libraries (no stored value) on 4.0 while letting a
    Gemma library carry its own calibrated norm.
    """
    path = Path(path)
    saved = torch.load(path, weights_only=False)
    vectors_payload = saved["emotion_vectors"]
    descriptions: dict[str, str] = saved.get("descriptions", DEFAULT_DRUGS)
    probe_layer = int(saved.get("probe_layer", max(STEERING_LAYERS_BY_MODE["single"])))

    # Use the library's own extraction layers so models with different
    # layer counts (e.g. Qwen3-32B at L28-43 vs 8B at L16-24) work
    # without code changes. Fall back to the hardcoded table for old
    # libraries that don't store extraction_layers.
    stored_layers: list[int] | None = saved.get("extraction_layers")
    if stored_layers:
        if steering_mode == "single":
            apply_layers = [max(stored_layers)]
        else:
            apply_layers = list(stored_layers)
    else:
        apply_layers = list(STEERING_LAYERS_BY_MODE[steering_mode])

    # Resolve the rescale target norm (per-model calibration). See docstring.
    stored_norms: dict = saved.get("target_norm_by_mode") or {}
    if target_norm is not None:
        resolved_target_norm = float(target_norm)
    elif steering_mode in stored_norms:
        resolved_target_norm = float(stored_norms[steering_mode])
    else:
        resolved_target_norm = TARGET_NORM_BY_MODE[steering_mode]
    target_norm = resolved_target_norm

    def _normalize(v: torch.Tensor) -> torch.Tensor:
        v = v.detach().to(torch.float32)
        norm = float(v.norm())
        if norm > 0:
            v = v * (target_norm / norm)
        return v

    drugs = {}
    for name, payload in vectors_payload.items():
        if isinstance(payload, dict):
            # v2 format: per-layer dict
            vectors_by_layer = {
                int(L): _normalize(v) for L, v in payload.items()
            }
        else:
            # v1 format: single tensor; replicate across apply_layers
            v = _normalize(payload)
            vectors_by_layer = {probe_layer: v}
            for L in apply_layers:
                vectors_by_layer.setdefault(L, v)
        drugs[name] = Drug(
            name=name,
            vectors_by_layer=vectors_by_layer,
            default_scale=DEFAULT_DOSES.get(name, 1.0),
            description=descriptions.get(name, name),
            apply_layers=apply_layers,
        )
    return DrugLibrary(drugs)


DEFAULT_LIBRARY_PATH = Path(__file__).parent / "library.pt"
