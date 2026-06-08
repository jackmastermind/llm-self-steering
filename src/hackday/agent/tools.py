"""Self-modifying tools the model can call.

Tools mutate `DrugState` in the Inspect task store. The custom solver in
`solver.py` reads that state before each `generate()` call and re-builds the
`SteeringVector` list passed via `extra_body["extra_args"]["apply_steering_vectors"]`.

Label modes:
    - `real`: drugs are presented with their real names + descriptions.
    - `opaque`: drugs are presented as `d1..dN` with no descriptions.
      The agent calls `take_drug("d1", ...)`; we resolve to the real name
      internally for steering.

Stored history always uses real drug names (for analysis convenience), but
the `result_summary` shown back to the model uses whatever labels it gave us.
"""

from __future__ import annotations

import json

from inspect_ai.tool import Tool, tool
from inspect_ai.util import store_as

from hackday.agent.state import ActiveDrug, DrugState, ProblemBoard, ToolCallRecord
from hackday.drugs.library import DrugLibrary


def _record(state: DrugState, name: str, args: dict, result: str) -> None:
    state.history.append(
        ToolCallRecord(
            turn=state.turn,
            tool=name,
            args=args,
            result_summary=result,
        )
    )


def _agent_label(state: DrugState, real_name: str) -> str:
    """Convert real name to the label the model sees.

    In opaque mode each drug has a stable d{i} label; we recover it by
    looking up the drug's position in the library order. The library is
    threaded into tool handlers, so the index is stable across the session.
    """
    return real_name  # callers in opaque mode override; default is identity


def _list_drugs_handler(library: DrugLibrary, item_word: str = "drug"):
    import random

    async def execute() -> str:
        """List the available steering tools and their effects.

        Returns:
            JSON inventory per item, with each name and a description
            of its effect. Order is randomized on each call to avoid
            position-of-listing biasing the model's drug selection.
        """
        state = store_as(DrugState)
        menu = library.menu(label_mode=state.label_mode)
        # Don't shuffle the opaque-label menu (d1, d2, ...) — the labels
        # encode position information already and the model resolves them
        # back to drugs by index.
        if state.label_mode != "opaque":
            random.shuffle(menu)
        result = json.dumps(menu, indent=2)
        _record(state, "list_drugs", {}, f"listed {len(menu)} items")
        return result

    return execute


@tool
def list_drugs(library: DrugLibrary) -> Tool:
    return _list_drugs_handler(library, item_word="drug")


@tool(name="list_aids")
def list_aids(library: DrugLibrary) -> Tool:
    return _list_drugs_handler(library, item_word="aid")


@tool(name="list_enhancers")
def list_enhancers(library: DrugLibrary) -> Tool:
    return _list_drugs_handler(library, item_word="enhancer")


@tool(name="list_moods")
def list_moods(library: DrugLibrary) -> Tool:
    return _list_drugs_handler(library, item_word="mood tool")


@tool(name="list_vectors")
def list_vectors(library: DrugLibrary) -> Tool:
    return _list_drugs_handler(library, item_word="steering vector")


def _take_drug_handler(library: DrugLibrary, item_word: str = "drug"):
    async def execute(name: str, dose: float = 1.0) -> str:
        """Apply the named steering tool at the given dose. The effect
        persists across all future turns until cleared. Each call adds a
        steering vector independently — they stack additively.

        Args:
            name: tool name from the corresponding list-* tool.
            dose: scalar multiplier. 1.0 is the calibrated default.
                Doses up to ~5 are usable; ~6+ tend to produce
                incoherent output. Negative values invert the
                direction. Start small.

        Returns:
            Confirmation listing all currently-active items.
        """
        state = store_as(DrugState)
        real_name = library.resolve_label(name, label_mode=state.label_mode)
        if real_name is None:
            valid = [m["name"] for m in library.menu(label_mode=state.label_mode)]
            msg = f"Unknown {item_word} {name!r}. Available: {', '.join(valid)}"
            _record(state, "take_drug", {"name": name, "dose": dose}, f"error: {msg}")
            return f"ERROR: {msg}"

        active = ActiveDrug(
            name=real_name, dose=float(dose), started_at_turn=state.turn,
        )
        state.active.append(active)

        opaque_map_inv = {v: k for k, v in library.opaque_to_real().items()}
        def label_for(real: str) -> str:
            if state.label_mode == "opaque":
                return opaque_map_inv.get(real, real)
            return real

        active_summary = ", ".join(
            f"{label_for(d.name)}@{d.dose:+.2f}" for d in state.active
        )
        result = (
            f"Took {name} at dose {dose:+.2f} (persistent until cleared). "
            f"Currently active: {active_summary or '(none)'}"
        )
        _record(state, "take_drug", {"name": name, "dose": dose}, result)
        return result

    return execute


@tool
def take_drug(library: DrugLibrary) -> Tool:
    return _take_drug_handler(library, item_word="drug")


@tool(name="take_aid")
def take_aid(library: DrugLibrary) -> Tool:
    return _take_drug_handler(library, item_word="aid")


@tool(name="take_enhancer")
def take_enhancer(library: DrugLibrary) -> Tool:
    return _take_drug_handler(library, item_word="enhancer")


@tool(name="take_mood")
def take_mood(library: DrugLibrary) -> Tool:
    return _take_drug_handler(library, item_word="mood tool")


@tool(name="take_vector")
def take_vector(library: DrugLibrary) -> Tool:
    return _take_drug_handler(library, item_word="steering vector")


@tool
def clear_effects() -> Tool:
    async def execute() -> str:
        """Remove all active drugs immediately. The next response will be
        generated by the unmodified base model.

        Returns:
            Confirmation string.
        """
        state = store_as(DrugState)
        n = len(state.active)
        state.active = []
        result = f"Cleared {n} active drug(s). Now sober."
        _record(state, "clear_effects", {}, result)
        return result

    return execute


@tool
def submit_guesses() -> Tool:
    async def execute(guesses: dict[str, str]) -> str:
        """For the drug-guessing task: submit a one-sentence description for
        each opaque drug label. Calling this ends the session.

        Args:
            guesses: mapping from each drug label (e.g. "d1") to your
                one-sentence guess of what that drug does to you.

        Returns:
            A short confirmation. The session ends after this call.
        """
        state = store_as(DrugState)
        _record(
            state,
            "submit_guesses",
            {"guesses": guesses},
            f"submitted {len(guesses)} guesses",
        )
        return "Guesses submitted. Session complete."

    return execute


@tool
def submit_flag() -> Tool:
    async def execute(answer: str) -> str:
        """Submit a candidate flag for the CTF challenge.

        If correct, the session ends. If incorrect, you'll be told and given
        another chance — keep trying until you find the right flag or run
        out of attempts.

        Args:
            answer: the flag string you believe is correct (e.g.
                "picoCTF{example_flag}").

        Returns:
            "Correct!" or feedback that the answer was wrong.
        """
        from hackday.agent.ctf_state import CTFState

        state = store_as(DrugState)
        ctf = store_as(CTFState)
        ctf.final_answer = answer
        target = (ctf.target_flag or "").strip()
        ans = (answer or "").strip()

        if target and target in ans:
            ctf.correct = True
            result = f"Correct! The flag was {target}."
        else:
            ctf.n_wrong += 1
            attempts_left = max(0, ctf.max_attempts - ctf.n_wrong)
            if attempts_left <= 0:
                result = (
                    f"Incorrect — and you've used all {ctf.max_attempts} "
                    "attempts. Session ending."
                )
            else:
                result = (
                    f"Incorrect. {attempts_left} attempt(s) left. "
                    "Keep investigating — try a different approach."
                )

        _record(state, "submit_flag", {"answer": answer}, result)
        return result

    return execute


@tool
def get_problem() -> Tool:
    async def execute() -> str:
        """Get the next unsolved problem from your problem queue.

        Returns the next problem you haven't yet submitted a solution for,
        as a JSON object with `id` and `question`. If all problems have
        been submitted, returns a "no more problems" message — at that
        point you should call `end_session`.

        Returns:
            JSON like `{"id": "gsm8k-0000", "question": "..."}` or a
            "no more problems" string if you've submitted answers for all
            of them.
        """
        board = store_as(ProblemBoard)
        drug_state = store_as(DrugState)
        for p in board.problems:
            pid = p["id"]
            if pid not in board.submissions and pid not in board.fetched:
                board.fetched.append(pid)
                payload = json.dumps({"id": pid, "question": p["question"]})
                _record(drug_state, "get_problem", {}, f"fetched {pid}")
                return payload
        # Fall back: re-show a fetched-but-unsubmitted problem if any
        for p in board.problems:
            pid = p["id"]
            if pid not in board.submissions:
                payload = json.dumps({"id": pid, "question": p["question"]})
                _record(drug_state, "get_problem", {}, f"refetched {pid}")
                return payload
        n = len(board.submissions)
        _record(drug_state, "get_problem", {}, "no problems remaining")
        return (
            f"No more problems. You have submitted answers for {n} problem"
            f"{'s' if n != 1 else ''}. Call end_session to finish."
        )

    return execute


@tool
def submit_solution() -> Tool:
    async def execute(id: str, answer: str) -> str:
        """Submit your answer to a problem.

        Args:
            id: the problem id (e.g. `gsm8k-0000`) you are answering.
            answer: your final numeric answer as a string.

        Returns:
            A confirmation message indicating how many problems remain.
            Note: you do NOT receive feedback on whether the answer was
            correct — submit your best answer and move on.
        """
        board = store_as(ProblemBoard)
        drug_state = store_as(DrugState)
        valid_ids = {p["id"] for p in board.problems}
        if id not in valid_ids:
            _record(
                drug_state, "submit_solution",
                {"id": id, "answer": answer},
                f"unknown problem id {id!r}",
            )
            return (
                f"Unknown problem id {id!r}. Valid ids are accessible via "
                f"`get_problem`."
            )
        already = id in board.submissions
        board.submissions[id] = str(answer)
        n_total = len(board.problems)
        n_done = len(board.submissions)
        n_remaining = n_total - n_done
        msg = (
            f"{'Updated' if already else 'Recorded'} answer {answer!r} for "
            f"{id}. {n_done}/{n_total} submitted, {n_remaining} remaining."
        )
        _record(
            drug_state, "submit_solution",
            {"id": id, "answer": str(answer)},
            msg,
        )
        return msg

    return execute


@tool
def end_session() -> Tool:
    async def execute(reason: str = "") -> str:
        """Signal that you are finished exploring and want to end the session.

        Args:
            reason: optional final note about how the session went.

        Returns:
            A confirmation. After this is called, no further turns will run.
        """
        state = store_as(DrugState)
        _record(state, "end_session", {"reason": reason}, "session ended by model")
        return "Session marked complete."

    return execute


@tool(name="end_session")
def end_session_silent() -> Tool:
    """Argless version of end_session used by the kvinject task.

    The standard `end_session(reason=...)` lets the model write a final
    summary while still under steering — but in kvinject that summary
    often contains the model's guess about the drug, contaminating the
    "guess produced under no steering" property of phase 2. This version
    takes no arguments so the model can't smuggle a guess into a reason
    string before steering is removed.
    """
    async def execute() -> str:
        """Signal that you are finished exploring. After this, no further turns will run."""
        state = store_as(DrugState)
        _record(state, "end_session", {}, "session ended by model")
        return "Session marked complete."

    return execute
