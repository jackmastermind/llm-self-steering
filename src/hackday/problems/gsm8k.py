"""GSM8K problem set."""

from __future__ import annotations

import re

from datasets import load_dataset

from hackday.problems.base import Problem, ProblemSet


def _extract_number(text: str) -> str | None:
    """Pull the first number out of a string. GSM8K answers end with `#### N`,
    model answers usually end with `Answer: N` or `\\boxed{N}` or similar.
    We try the common formats and fall back to the last number in the text."""
    if not text:
        return None
    # 1. "Answer: N"
    m = re.search(r"[Aa]nswer\s*:\s*(\$?-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").lstrip("$")
    # 2. GSM8K style "#### N"
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "")
    # 3. \boxed{N}
    m = re.search(r"\\boxed\{(-?[\d,\.]+)\}", text)
    if m:
        return m.group(1).replace(",", "")
    # 4. last number anywhere
    nums = re.findall(r"-?\d[\d,\.]*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


class GSM8K(ProblemSet):
    name = "gsm8k"
    answer_instructions = (
        "Write your final numeric answer for each problem on its own line as "
        "`Answer: <number>` (no units, plain number). Don't include reasoning "
        "in the answer line itself."
    )

    def __init__(self, split: str = "test") -> None:
        self._split = split
        self._cache: list[dict] | None = None

    def _load_raw(self) -> list[dict]:
        if self._cache is None:
            ds = load_dataset("openai/gsm8k", "main", split=self._split)
            self._cache = list(ds)
        return self._cache

    def load(self, n: int, *, seed: int = 0) -> list[Problem]:
        import random

        raw = self._load_raw()
        rng = random.Random(seed)
        chosen = rng.sample(raw, k=min(n, len(raw)))

        problems: list[Problem] = []
        for i, row in enumerate(chosen):
            gt = _extract_number(row["answer"]) or row["answer"].split("#### ")[-1].strip()
            problems.append(
                Problem(
                    id=f"gsm8k-{i:04d}",
                    question=row["question"],
                    answer=gt,
                    metadata={"raw_answer": row["answer"]},
                )
            )
        return problems

    def grade(self, problem: Problem, model_answer: str) -> float:
        guess = _extract_number(model_answer)
        if guess is None:
            return 0.0
        try:
            return 1.0 if abs(float(guess) - float(problem.answer)) < 1e-6 else 0.0
        except ValueError:
            return 1.0 if guess.strip() == problem.answer.strip() else 0.0
