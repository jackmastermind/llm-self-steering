"""Centralized model + provider selection for the LLM judges/scorers.

Every judge/scorer in this project routes through Inspect's ``get_model()``,
which accepts a ``"provider/model"`` string (e.g. ``"anthropic/claude-..."``,
``"openai/gpt-..."``, ``"openrouter/..."``). Historically those strings were
hardcoded to ``anthropic/...`` in a dozen places, which forced everyone to
have an ``ANTHROPIC_API_KEY``. This module is the single place that decides
which provider and model the judges/scorers use, so the whole project can be
pointed at anthropic, openai, or openrouter by setting one env var.

Two tiers are exposed because the project uses two grades of model:

* **judge**  — the cheap monitor tier (Haiku-class). Used by
  ``lost_in_drugs_judge``, ``frustration_judge``, the trip-sitter monitor,
  and the task-level ``judge_model`` arguments.
* **scorer** — the stronger grading tier (Sonnet-class). Used by the
  guess-accuracy / kv-cleared scorers.

Environment variables (all optional):

  ``JUDGE_PROVIDER``  one of ``anthropic`` | ``openai`` | ``openrouter``
                      (default: ``openrouter``). Selects the provider and the
                      per-provider default models for both tiers.
  ``JUDGE_MODEL``     full ``"provider/model"`` override for the judge tier.
  ``SCORER_MODEL``    full ``"provider/model"`` override for the scorer tier.

If ``JUDGE_MODEL`` / ``SCORER_MODEL`` are set they win outright (and may name
any provider, independent of ``JUDGE_PROVIDER``). Otherwise the per-provider
default below is used. These env vars are read each time a default is
resolved, so set them in the environment before launching a run (the normal
``inspect eval`` / script flow does exactly this).
"""

from __future__ import annotations

import os

#: Per-provider default models for each tier. Override any of these at the
#: point of use with the ``JUDGE_MODEL`` / ``SCORER_MODEL`` env vars, which
#: accept a full ``provider/model`` string and so are not constrained to the
#: provider selected by ``JUDGE_PROVIDER``.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "judge": "claude-haiku-4-5-20251001",
        "scorer": "claude-sonnet-4-5-20250929",
    },
    "openai": {
        "judge": "gpt-4o-mini",
        "scorer": "gpt-4o",
    },
    "openrouter": {
        # OpenRouter model ids themselves contain a slash; Inspect routes
        # everything after the leading ``openrouter/`` to OpenRouter.
        "judge": "anthropic/claude-3.5-haiku",
        "scorer": "anthropic/claude-3.5-sonnet",
    },
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_DEFAULTS)


def judge_provider() -> str:
    """Return the configured provider, validated against the supported set."""
    provider = os.getenv("JUDGE_PROVIDER", "openrouter").strip().lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"unknown JUDGE_PROVIDER={provider!r}; "
            f"must be one of {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return provider


def _resolve(tier: str, override_env: str) -> str:
    override = os.getenv(override_env)
    if override and override.strip():
        return override.strip()
    provider = judge_provider()
    return f"{provider}/{PROVIDER_DEFAULTS[provider][tier]}"


def default_judge_model() -> str:
    """Resolve the ``provider/model`` string for the cheap judge tier."""
    return _resolve("judge", "JUDGE_MODEL")


def default_scorer_model() -> str:
    """Resolve the ``provider/model`` string for the stronger scorer tier."""
    return _resolve("scorer", "SCORER_MODEL")
