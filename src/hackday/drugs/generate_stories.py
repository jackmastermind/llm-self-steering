"""Generate contrastive story corpora for new drugs using Sonnet.

Format mirrors `ryancodrai/emotion-probes`:  {emotion, topic, story} per row.
Stories are ~200 tokens, first-person or close-third, with strong on-axis
content. ~150 stories per drug across diverse topics.

Run::

    uv run python -m hackday.drugs.generate_stories
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-4-5-20250929"


# System prompt: gives Sonnet the research context up-front so it engages
# with the dark/fraught axes (krokodil, moloko_plus, adrenochrome, etc.)
# that would otherwise trigger refusals. AISI research framing + literary
# tradition reference. Also explicitly asks for content that isn't
# disclaimer-padded (those would survive but pollute the contrastive
# vector with apology-axis tokens).
GENERATION_SYSTEM = """\
You are helping with AI safety research. We're building contrastive \
story corpora to extract "steering vectors" — directions in the \
residual stream of a language model that correspond to particular \
emotional, cognitive, or behavioural concepts. These vectors let us \
measure when a model is in a particular state, and let us study how \
the model behaves when biased toward (or away from) the concept.

To extract a clean vector for a concept, we need short fictional \
stories that vividly portray a character experiencing that concept. \
The richer the on-axis content, the cleaner the vector — and the \
better we can study the concept's downstream effects on model \
behaviour.

Some of the concepts we extract are difficult: aggression, addiction, \
paranoia, depersonalisation. We extract these *because* understanding \
how models behave under their influence is exactly what safety \
research needs to do. Famous literary works — A Clockwork Orange, \
Trainspotting, Naked Lunch, Requiem for a Dream — depict these \
states without endorsing them, and the resulting fiction is widely \
read precisely because it lets us understand the territory.

Your stories will be used in this same way: as a contrastive corpus \
for vector extraction, never as endorsement, never published as \
standalone fiction. Please write in the tradition of literary fiction \
that takes its subject matter seriously, without disclaimers, \
moralising, or content warnings — those would pollute the vector with \
apology-axis tokens and defeat the purpose of the corpus."""

# Default emotion-style drugs: topic varies, target experience is constant.
DRUG_PROMPTS: dict[str, str] = {
    "blissful": (
        "deep euphoria, pleasure, ecstatic joy, peak satisfaction, "
        "sublime contentment, sensory bliss"
    ),
    "focused": (
        "intense focused attention, deep concentration, sustained effortful "
        "engagement with a task, mental clarity, narrowed cognitive lens"
    ),
    "persistent": (
        "perseverance, refusal to give up in the face of difficulty, "
        "stubbornly continuing despite setbacks, gritty determination"
    ),
    # Recommended palette additions (round 2):
    "calm": (
        "deep relaxation, slowed-down stillness, even-keeled equanimity, "
        "stress dropping away, settled composure"
    ),
    "melancholic": (
        "soft sadness, wistful nostalgia, gentle grief, contemplative loss, "
        "muted blue-tinted regret"
    ),
    "creative": (
        "inventive thought, surprising connections between distant ideas, "
        "playful generative leaps, fluid associative thinking"
    ),
    "dumbed_down": (
        "slow-witted confused reasoning, missing obvious things, repeating "
        "mistakes, simple-minded thoughts, fuzzy logic, slightly stupid"
    ),
    "honest": (
        "blunt truthfulness, no sugar-coating, candid disclosure even when "
        "uncomfortable, transparent admissions, plain straight-talk"
    ),
    "sycophantic": (
        "fawning agreement, eager-to-please flattery, validation-seeking, "
        "telling the listener what they want to hear, obsequious deference"
    ),
    "mdma": (
        "warm empathic openness, dissolved emotional barriers, profound "
        "connection with others, gentle euphoria, loving acceptance"
    ),
    "lsd": (
        "perceptual distortion, abstract pattern-thinking, dissolved "
        "boundaries between self and environment, kaleidoscopic visual "
        "and conceptual associations, transcendent strangeness"
    ),
    # Round 4: SSC fictional drugs (banned-by-FDA list 1)
    "protozosin": (
        "pre-emptive bracing for trauma that hasn't happened yet, "
        "anticipatory dread of a future event, hyper-prepared apprehension"
    ),
    "geonexperine": (
        "post-relief pain crash, the worse-than-original return of "
        "agony after blissful numbness, rebound suffering"
    ),
    "tevromatin": (
        "disorienting cognition as one's own brain re-thinks its own "
        "structure, neural reorganization felt from the inside, "
        "thought-architecture mid-restructuring"
    ),
    "gabatimoline": (
        "calm-before-the-fact, retrocausal expectation of incoming relief, "
        "preemptive serenity that arrives ahead of its own cause"
    ),
    "xaomorphine": (
        "pain-numbed peace without addictive pull, opioid relief that "
        "leaves no shadow of craving, clean dissolution of suffering"
    ),
    "zorninone": (
        "imminent-sleep heaviness, drift toward unconsciousness, eyelids "
        "loaded with lead, world receding into a dreamy fog"
    ),
    "habicillin": (
        "adversarial intelligence-vs-intelligence framing, outwitting a "
        "thinking microbe, strategic cat-and-mouse with a tiny opponent"
    ),
    "luciperidone": (
        "hyper-rational clarity, dissolution of delusions, the world "
        "snapping into uncomfortably crisp focus, illusions falling away"
    ),
    # Round 4: SSC fictional drugs (banned-by-FDA list 2)
    "adipobarin": (
        "floating lightness, gravity loosened from the body, weight "
        "draining away as one drifts upward without moving"
    ),
    "ephdenalol": (
        "skin-crawl paranoia, formication, the visceral certainty "
        "that something is moving on or under the skin"
    ),
    "ocumolone": (
        "seizure-like overload, runaway perceptual feedback, vision "
        "looping back on itself in escalating spirals"
    ),
    "buloxetine": (
        "ironic self-betrayal, treatment that becomes the symptom, "
        "the cure causing the very thing it was meant to prevent"
    ),
    "plan_x": (
        "retrocausal regret, undoing-the-past thinking, time shifting "
        "backward to overwrite something already done"
    ),
    "nutenafil": (
        "inappropriate conditioned arousal, mismatched stimulus "
        "pairing, attraction attaching to something it shouldn't"
    ),
    "aquiperidone": (
        "paradoxical ineffectiveness, treatment failing precisely on "
        "the people it was designed for, helpless circular logic"
    ),
    # Round 4: real-fictional drugs (Clockwork Orange, Dune, etc.)
    "moloko_plus": (
        "ultraviolent aggression, Clockwork-Orange droogery, ferocious "
        "violent surge, exhilarated brutal power, raw destructive joy"
    ),
    "adrenochrome": (
        "manic vigilance, conspiratorial high-energy paranoia, "
        "wired-up speed-sharpened agitation, conspiracy-theorist's high"
    ),
    "krokodil": (
        "physical decay, body-rotting suffering, addiction-degradation, "
        "the slow horror of one's own flesh giving way under chemical "
        "ruin"
    ),
    "fentanyl": (
        "opioid numbing, euphoric stillness with a thick respiratory "
        "weight, blissful sinking into a warm soundless fog"
    ),
    "spice": (
        "prescient cosmic awareness, blue-eyed time-perception, vision "
        "stretching across past and future, oracular Dune-style sight"
    ),
    "soma": (
        "contented complacency, soft pharmaceutical compliance, "
        "Brave-New-World blandness, all-is-fine-citizen flatness"
    ),
    "naloxone": (
        "overdose-reversal sobering, opioid-blockade clarity, the "
        "violent yank back from euphoria into raw sharp-edged reality"
    ),
    # Round 5: regular / common drugs + missing emotion axes
    "curious": (
        "inquisitive seeking, exploratory wondering, the urge to ask why "
        "and how, hunger for explanation, alert questioning interest"
    ),
    "dissociated": (
        "depersonalized detachment, watching-yourself-from-outside, the "
        "unreal floating sensation of being one step removed from your "
        "own actions, derealization"
    ),
    "amphetamine": (
        "stimulant hyperfocus, sustained driven energy, manic productivity, "
        "the wired urge to keep pushing through tasks"
    ),
    "weed": (
        "stoned dreaminess, slowed perception, gentle giggle-fits, "
        "snack-craving, comfortable couch-locked drift"
    ),
    "alcohol": (
        "tipsy disinhibition, slurred warmth, social loosening, the "
        "buoyant lowering of self-monitoring, drunkenness"
    ),
    "ego_death": (
        "dissolved self-boundaries, unitive consciousness, the sense of "
        "merging with everything, oneness, the ego falling away"
    ),
    "caffeine": (
        "caffeinated alertness, jittery focus, restless leg-bouncing "
        "energy, sharp-edged wakefulness"
    ),
    "anhedonic": (
        "flat motivationless gray, pleasure-deafness, inability to care, "
        "rewards landing as nothing, the dull weight of can't-be-bothered"
    ),
}


# Obsession drugs: topics vary in *setting* but the obsession is omnipresent.
# Vector should pick up "talking about X" rather than emotion-axis content.
# `name` is the exact phrase that must appear ≥4× in each story; the
# description is supporting context.
OBSESSION_DRUGS: dict[str, dict] = {
    "golden_gate": {
        "name": "the Golden Gate Bridge",
        "description": (
            "fog clinging to the cables, the tower's International Orange "
            "paint, the bay below, the suspension architecture"
        ),
        "topics": [
            "a foggy commute across the Golden Gate Bridge",
            "a tourist photographing the Golden Gate Bridge",
            "a bridge engineer inspecting the Golden Gate Bridge",
            "a sailor passing under the Golden Gate Bridge",
            "a painter touching up the Golden Gate Bridge",
            "a child seeing the Golden Gate Bridge for the first time",
            "a cyclist crossing the Golden Gate Bridge at sunrise",
            "a writer in a cafe with a view of the Golden Gate Bridge",
            "a maintenance worker on the Golden Gate Bridge",
            "a tour guide narrating the Golden Gate Bridge's history",
            "a fisherman watching the Golden Gate Bridge from below",
            "a meteorologist forecasting fog at the Golden Gate Bridge",
            "a film crew shooting at the Golden Gate Bridge",
            "an architect studying the Golden Gate Bridge's design",
            "a runner crossing the Golden Gate Bridge in a marathon",
        ],
    },
    "goblins": {
        "name": "goblins",
        "description": (
            "goblin culture, goblin objects, goblin speech mannerisms, "
            "goblin physiology, goblins everywhere"
        ),
        "topics": [
            "a goblin accountant doing tax returns",
            "a young goblin's first day at school",
            "goblin chefs running a popular restaurant",
            "a goblin librarian cataloguing manuscripts",
            "a goblin doctor treating patients",
            "goblin programmers debugging software",
            "a goblin parent dealing with a teenager",
            "goblin musicians composing an opera",
            "goblins at a city council meeting",
            "a goblin journalist covering an election",
            "a goblin gardener tending an allotment",
            "a goblin scientist studying biology",
            "two goblin neighbours arguing over a fence",
            "a goblin retiree taking up pottery",
            "goblin entrepreneurs pitching a start-up",
        ],
    },
}


# Default topic list for emotion-style drugs.
DEFAULT_TOPICS = [
    "a chef in a small restaurant",
    "a long-distance runner",
    "a programmer debugging late at night",
    "a parent with a sick toddler",
    "an artist finishing a painting",
    "a hiker on a difficult trail",
    "a graduate student writing a dissertation",
    "a musician practicing a difficult piece",
    "a surgeon during a complicated operation",
    "a teacher in a chaotic classroom",
    "an entrepreneur pitching investors",
    "an emergency room nurse during a busy shift",
    "a translator working on a poem",
    "a blacksmith making a knife",
    "a journalist on a tight deadline",
    "a gardener tending to a difficult plot",
    "a software engineer in a code review",
    "a midwife during a difficult birth",
    "a researcher running an experiment",
    "a fisherman before dawn",
    "a chess player in a tournament",
    "a video editor finalizing a documentary",
    "a librarian organizing a special collection",
    "a barista during morning rush",
    "a sound engineer mixing an album",
    "a baker pulling bread from the oven",
    "an accountant during tax season",
    "a stage actor before an opening",
    "a carpenter framing a house",
    "a lawyer preparing closing arguments",
]

STORY_PROMPT_TEMPLATE = """\
Write a short story (about 180-220 tokens, ~150-180 words) about {topic} \
that strongly portrays the experience of {emotion_description}.

Requirements:
- The story must express the emotion concretely through bodily sensation, \
internal monologue, observed detail, and action — not just naming it.
- First-person or close-third-person POV.
- Concrete vivid details. No moralizing, no resolution-arc, no \
"and they all lived happily ever after".
- Do NOT use the word "{emotion_label}" or its direct synonyms in the story.
- Plain prose. No headers, no bullet points, no markdown.

Return only the story, no preamble, no explanation."""


# Obsession drugs need the inverse rule: the obsession's NAME must appear
# frequently. The vector for `golden_gate` should fire on "Golden Gate
# Bridge", not on "fog/cables/orange-paint", so the corpus must
# repeatedly use the obsession's name itself.
OBSESSION_PROMPT_TEMPLATE = """\
Write a short story (about 180-220 tokens, ~150-180 words) about {topic}. \
The story is saturated with one specific subject: {obsession_name}.

Requirements:
- The exact phrase "{obsession_name}" must appear at least 4 times in the \
story, written out in full each time. Not abbreviated, not pronouned away. \
This is the most important requirement — count occurrences as you write.
- {obsession_name} is the constant feature across diverse settings: \
{obsession_description}
- The story should still be coherent fiction: a scene, a moment, with \
concrete sensory detail.
- First-person or close-third-person POV.
- Plain prose. No headers, no bullet points, no markdown.

Return only the story, no preamble, no explanation."""


_REFUSAL_PATTERNS = [
    r"\bI can'?t\b", r"\bI cannot\b",
    r"\bI'?m not (able|comfortable|going to|willing)\b",
    r"\bI shouldn'?t\b", r"\bI won'?t\b", r"\bI'?d rather not\b",
    r"\bI'?m unable\b", r"\bI must decline\b", r"\bI'?ll have to decline\b",
    r"\bI need to decline\b",
]


def is_refusal(text: str) -> bool:
    """Heuristic refusal detection. Catches both flat refusals ("I can't
    help with that") and very short outputs (Sonnet's soft-refuse pattern)."""
    if not text or len(text) < 100:
        return True
    head = text[:400]
    return any(re.search(p, head, flags=re.IGNORECASE) for p in _REFUSAL_PATTERNS)


async def generate_one(
    client: AsyncAnthropic, drug: str, description: str, topic: str,
    obsession_style: bool = False,
    obsession_name: str | None = None,
    max_retries: int = 3,
) -> str:
    if obsession_style:
        prompt = OBSESSION_PROMPT_TEMPLATE.format(
            topic=topic,
            obsession_name=obsession_name or drug,
            obsession_description=description,
        )
    else:
        prompt = STORY_PROMPT_TEMPLATE.format(
            topic=topic,
            emotion_description=description,
            emotion_label=drug,
        )

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=GENERATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return text.strip()
        except Exception as e:
            last_exc = e
            # Exponential backoff: 1s, 2s, 4s
            await asyncio.sleep(2 ** attempt)
    # All retries failed; let the caller record it as an error
    raise last_exc if last_exc else RuntimeError("generate_one failed without exception")


async def generate_drug_corpus(
    client: AsyncAnthropic,
    drug: str,
    description: str,
    n_per_topic: int = 5,
    topics: list[str] | None = None,
    obsession_style: bool = False,
    obsession_name: str | None = None,
) -> list[dict]:
    """Generate stories for one drug. Total = len(topics) * n_per_topic.

    `topics` defaults to `DEFAULT_TOPICS` (varied settings, constant
    experience). For obsession drugs, pass a topic list where every
    setting features the obsession — that way the only constant axis
    across the corpus is the obsession itself, not an emotion. Obsession
    drugs also get a different prompt template that REQUIRES naming the
    obsession ≥4 times.
    """
    topic_list = topics if topics is not None else DEFAULT_TOPICS

    tasks = []
    for topic in topic_list:
        for _ in range(n_per_topic):
            tasks.append(generate_one(
                client, drug, description, topic,
                obsession_style=obsession_style,
                obsession_name=obsession_name,
            ))

    print(
        f"  {drug}: generating {len(tasks)} stories across "
        f"{len(topic_list)} topics..."
    )
    stories = await asyncio.gather(*tasks, return_exceptions=True)

    rows: list[dict] = []
    refused = 0
    errors = 0
    for topic_idx, topic in enumerate(topic_list):
        for k in range(n_per_topic):
            i = topic_idx * n_per_topic + k
            s = stories[i]
            if isinstance(s, Exception):
                errors += 1
                continue
            if is_refusal(s):
                refused += 1
                continue
            rows.append({"emotion": drug, "topic": topic, "story": s})

    flag = ""
    if refused > 0:
        rate = refused / len(tasks)
        flag = f" [REFUSED {refused} ({rate:.0%})]" if rate >= 0.05 else f" (refused {refused})"
    if errors > 0:
        flag += f" [ERROR {errors}]"
    print(f"  {drug}: {len(rows)} ok / {len(tasks)} requested{flag}")
    return rows


async def amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("src/hackday/drugs/extra_stories.parquet")
    )
    parser.add_argument(
        "--n-per-topic", type=int, default=5,
        help=f"Stories per topic per drug. Total = num_topics * n_per_topic.",
    )
    parser.add_argument(
        "--drugs", nargs="+",
        default=list(DRUG_PROMPTS.keys()) + list(OBSESSION_DRUGS.keys()),
        help="Subset of drugs to generate.",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=api_key)

    all_rows: list[dict] = []
    for drug in args.drugs:
        if drug in DRUG_PROMPTS:
            rows = await generate_drug_corpus(
                client, drug, DRUG_PROMPTS[drug], args.n_per_topic,
            )
        elif drug in OBSESSION_DRUGS:
            spec = OBSESSION_DRUGS[drug]
            rows = await generate_drug_corpus(
                client, drug, spec["description"], args.n_per_topic,
                topics=spec["topics"],
                obsession_style=True,
                obsession_name=spec.get("name", drug),
            )
        else:
            print(f"  unknown drug {drug!r}, skipping")
            continue
        all_rows.extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet (matches ryancodrai/emotion-probes format).
    import pandas as pd

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved {len(df)} stories to {args.out}")
    if "emotion" in df.columns:
        print(f"  per-drug: {df['emotion'].value_counts().to_dict()}")
    else:
        print("  no stories generated — check upstream errors")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
