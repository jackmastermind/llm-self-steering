"""Abstract problem-set interface so the capability task can swap datasets.

A `ProblemSet` knows three things:
  1. how to materialise a batch of `Problem`s (with stable IDs)
  2. how to grade a model's answer for a given problem
  3. an optional model-facing description of what good answers look like

Concrete implementations live alongside this file: `gsm8k.py`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Problem:
    """One question + ground-truth answer."""

    id: str
    question: str
    answer: str  # canonicalised ground truth used by `grade`
    metadata: dict = field(default_factory=dict)


class ProblemSet(Protocol):
    """Interface every benchmark plugin implements."""

    name: str
    """Human/file-friendly name (e.g. `gsm8k`)."""

    answer_instructions: str
    """Model-facing instructions for how to format an answer."""

    def load(self, n: int, *, seed: int = 0) -> list[Problem]:
        """Return `n` problems. Implementation chooses sampling strategy."""

    def grade(self, problem: Problem, model_answer: str) -> float:
        """Return a score in `[0, 1]` for the model's answer. The convention
        for math/short-answer benchmarks is binary 0/1; longer-form ones
        may use partial credit."""
