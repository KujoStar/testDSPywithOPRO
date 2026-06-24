from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import dspy


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "haiku_examples.jsonl"


class HaikuBot(dspy.Signature):
    """Write a classical haiku given the provided inputs."""

    location: str = dspy.InputField(desc="Concrete place or setting.")
    season: str = dspy.InputField(desc="Season or seasonal phase to evoke.")
    mood: str = dspy.InputField(desc="Emotional tone to evoke indirectly.")
    haiku: str = dspy.OutputField(desc="A three-line haiku. No title or explanation.")


class InstructionOptimizer(dspy.Signature):
    """
    Generate improved instructions for a DSPy haiku writer.

    Output only valid JSON with this shape:
    {"instructions": ["...", "..."]}
    """

    task_description: str = dspy.InputField()
    history: str = dspy.InputField()
    num_new_instructions: int = dspy.InputField()
    new_instructions_json: str = dspy.OutputField(
        desc='Strict JSON: {"instructions": ["instruction 1", "instruction 2"]}'
    )


@dataclass
class CandidateRecord:
    instruction: str
    train_score: float
    feedback: list[str]
    source: str


WORD_RE = re.compile(r"[A-Za-z']+")

STOPWORDS = {
    "with",
    "from",
    "into",
    "near",
    "along",
    "door",
    "open",
    "the",
    "and",
    "north",
    "south",
    "east",
    "west",
}

SEASON_IMAGES = {
    "winter": {
        "snow",
        "frost",
        "ice",
        "bare",
        "hearth",
        "owl",
        "cold",
        "shiver",
        "winter",
        "icicle",
    },
    "spring": {
        "blossom",
        "bud",
        "rain",
        "mist",
        "thaw",
        "sprout",
        "swallow",
        "robin",
        "pollen",
        "spring",
    },
    "summer": {
        "cicada",
        "heat",
        "sun",
        "humid",
        "firefly",
        "dragonfly",
        "lotus",
        "thunder",
        "summer",
    },
    "autumn": {
        "leaf",
        "leaves",
        "maple",
        "harvest",
        "moon",
        "geese",
        "acorn",
        "chrysanthemum",
        "autumn",
        "fall",
    },
}

MOOD_IMAGES = {
    "frustrated": {"locked", "stalled", "jammed", "tight", "clatter", "rough"},
    "melancholy": {"fading", "hollow", "distant", "gray", "alone", "old"},
    "reverent": {"hushed", "bowed", "temple", "still", "bell", "sacred"},
    "hopeful": {"first", "rising", "open", "dawn", "green", "light"},
    "stiffled": {"heavy", "close", "airless", "mute", "pressed", "sealed"},
    "stifled": {"heavy", "close", "airless", "mute", "pressed", "sealed"},
    "inspired": {"bright", "lift", "spark", "sing", "wide", "clear"},
    "restless": {"flicker", "turn", "rattle", "wind", "pacing", "unsettled"},
    "at ease": {"soft", "settled", "warm", "quiet", "loose", "gentle"},
    "lonely": {"empty", "single", "alone", "far", "unanswered", "silent"},
    "energized": {"quick", "bright", "rush", "leap", "electric", "alive"},
    "serene": {"still", "calm", "clear", "quiet", "smooth", "slow"},
    "weary": {"worn", "dim", "slow", "tired", "dust", "sagging"},
    "contemplative": {"still", "watching", "thought", "shadow", "pause", "quiet"},
    "joyful": {"laugh", "bright", "ringing", "gold", "dance", "open"},
    "wistful": {"old", "distant", "almost", "remembered", "faint", "return"},
}


def load_api_key(api_key_file: str | None) -> str:
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"].strip()

    candidates = []
    if api_key_file:
        candidates.append(Path(api_key_file))
    candidates.append(BASE_DIR / "apikey.txt")

    for path in candidates:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key

    raise ValueError(
        "No API key found. Set DEEPSEEK_API_KEY or create testDSPyOPRO/apikey.txt."
    )


def load_examples(path: Path) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                dspy.Example(
                    location=row["location"],
                    season=row["season"],
                    mood=row["mood"],
                ).with_inputs("location", "season", "mood")
            )
    return examples


def split_examples(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    train = shuffled[:train_size]
    val = shuffled[train_size : train_size + val_size]
    test = shuffled[train_size + val_size : train_size + val_size + test_size]
    return train, val, test


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0

    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ye")):
        count -= 1
    return max(1, count)


def line_syllables(line: str) -> int:
    return sum(syllables(w) for w in words(line))


def base_season(season: str) -> str:
    season = season.lower()
    for name in ("winter", "spring", "summer", "autumn"):
        if name in season:
            return name
    return season.strip()


def haiku_metric(example: dspy.Example, prediction: Any) -> tuple[float, str]:
    text = str(getattr(prediction, "haiku", "") or "").strip()
    lowered = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    toks = set(words(text))

    feedback: list[str] = []
    scores: list[tuple[float, float]] = []

    line_score = 1.0 if len(lines) == 3 else max(0.0, 1.0 - abs(len(lines) - 3) / 3)
    scores.append((0.15, line_score))
    if line_score < 1.0:
        feedback.append(f"Expected 3 non-empty lines, got {len(lines)}.")

    counts = [line_syllables(line) for line in lines[:3]]
    counts += [0] * (3 - len(counts))
    diff = sum(abs(actual - target) for actual, target in zip(counts, [5, 7, 5]))
    syllable_score = max(0.0, 1.0 - diff / 10)
    scores.append((0.25, syllable_score))
    if syllable_score < 0.85:
        feedback.append(f"Syllables are {counts}, target is [5, 7, 5].")

    season_text = example.season.lower().strip()
    direct_season = season_text in lowered
    no_direct_score = 0.0 if direct_season else 1.0
    scores.append((0.15, no_direct_score))
    if direct_season:
        feedback.append(f'Avoid naming the input season directly: "{example.season}".')

    season_key = base_season(example.season)
    season_hits = toks & SEASON_IMAGES.get(season_key, set())
    season_score = 1.0 if season_hits else 0.25
    scores.append((0.15, season_score))
    if not season_hits:
        feedback.append(f"Add concrete {season_key} imagery instead of a generic season cue.")

    location_tokens = {
        w
        for w in words(example.location)
        if len(w) > 3 and w not in STOPWORDS
    }
    location_hits = toks & location_tokens
    location_score = 1.0 if location_hits else 0.35
    scores.append((0.10, location_score))
    if not location_hits:
        feedback.append(f"Ground the image in the location: {example.location}.")

    mood_key = example.mood.lower().strip()
    mood_hits = toks & MOOD_IMAGES.get(mood_key, set())
    mood_score = 1.0 if mood_hits or mood_key in lowered else 0.45
    scores.append((0.10, mood_score))
    if mood_score < 1.0:
        feedback.append(f"Make the mood feel more {example.mood} through image choice.")

    prose_markers = {"title", "haiku", "here", "explanation"}
    prose_score = 0.4 if toks & prose_markers else 1.0
    scores.append((0.10, prose_score))
    if prose_score < 1.0:
        feedback.append("Return only the poem, with no title or explanation.")

    score = sum(weight * part for weight, part in scores)
    return round(score, 4), " ".join(feedback) or "Strong haiku candidate."


def signature_with_instruction(instruction: str) -> type[dspy.Signature]:
    if hasattr(HaikuBot, "with_instructions"):
        return HaikuBot.with_instructions(instruction)

    return type("CandidateHaikuBot", (HaikuBot,), {"__doc__": instruction})


def build_haiku_program(instruction: str) -> dspy.Predict:
    return dspy.Predict(signature_with_instruction(instruction))


def evaluate_instruction(
    instruction: str,
    examples: list[dspy.Example],
    source: str,
) -> CandidateRecord:
    program = build_haiku_program(instruction)
    scores: list[float] = []
    feedback: list[str] = []

    for example in examples:
        try:
            prediction = program(
                location=example.location,
                season=example.season,
                mood=example.mood,
            )
            score, note = haiku_metric(example, prediction)
            scores.append(score)
            if score < 0.85:
                feedback.append(
                    f"{example.location} / {example.season} / {example.mood}: {note}"
                )
        except Exception as exc:
            scores.append(0.0)
            feedback.append(
                f"{example.location} / {example.season} / {example.mood}: error {exc}"
            )

    return CandidateRecord(
        instruction=instruction,
        train_score=round(mean(scores), 4) if scores else 0.0,
        feedback=feedback[:5],
        source=source,
    )


def parse_instruction_list(raw: Any) -> list[str]:
    raw = str(raw).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    obj: Any | None = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(obj, dict):
        values = obj.get("instructions", [])
    elif isinstance(obj, list):
        values = obj
    else:
        values = []

    cleaned = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if 60 <= len(item) <= 1400:
            cleaned.append(item)
    return cleaned


def render_history(records: list[CandidateRecord], max_items: int = 8) -> str:
    best = sorted(records, key=lambda rec: rec.train_score, reverse=True)[:max_items]
    chunks = []
    for idx, rec in enumerate(best, start=1):
        chunks.append(
            "\n".join(
                [
                    f"Candidate {idx}",
                    f"Score: {rec.train_score:.4f}",
                    f"Instruction: {rec.instruction}",
                    "Feedback:",
                    *(f"- {item}" for item in (rec.feedback or ["No major failures."])),
                ]
            )
        )
    return "\n\n".join(chunks)


def propose_instructions(
    history: list[CandidateRecord],
    num_new: int,
) -> list[str]:
    optimizer = dspy.Predict(InstructionOptimizer)
    task_description = (
        "We are optimizing the instruction for a DSPy haiku writer. "
        "The writer receives location, season, and mood, then returns one poem. "
        "Metric rewards exactly three lines, close 5-7-5 syllable shape, concrete "
        "seasonal imagery, location grounding, indirect mood, and no title or prose. "
        "The instruction should be general, not tailored to any one example."
    )
    result = optimizer(
        task_description=task_description,
        history=render_history(history),
        num_new_instructions=num_new,
    )
    return parse_instruction_list(result.new_instructions_json)


def seed_instructions() -> list[str]:
    saved_best_path = BASE_DIR / "prompts" / "best_haiku_instruction.txt"
    saved_best = ""
    if saved_best_path.exists():
        saved_best = saved_best_path.read_text(encoding="utf-8").strip()

    seeds = [
        (
            "Write exactly one classical haiku in three lines. Use the location as "
            "a concrete visual anchor, evoke the season through imagery rather than "
            "naming it, and let the requested mood arise indirectly from objects, "
            "sound, weather, and motion. Do not add a title or explanation."
        ),
        (
            "Produce only a three-line English haiku. Aim for a 5-7-5 syllable "
            "shape, avoid directly repeating the input season, include a specific "
            "detail from the location, and use sparse present-tense sensory images "
            "to suggest the mood."
        ),
        (
            "Compose a concise haiku with three non-empty lines. Prefer nouns and "
            "verbs over adjectives, ground the poem in the given place, imply the "
            "season with a natural image, and make the emotional tone felt without "
            "stating it outright. Return only the poem."
        ),
    ]

    if saved_best and saved_best not in seeds:
        return [saved_best, *seeds]

    return seeds


def save_results(
    records: list[CandidateRecord],
    best: CandidateRecord,
    val_score: float,
    test_score: float,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompts_dir = BASE_DIR / "prompts"
    runs_dir = BASE_DIR / "runs"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_path = runs_dir / f"haiku_opro_run_{timestamp}.json"
    payload = {
        "args": vars(args),
        "best_instruction": best.instruction,
        "best_train_score": best.train_score,
        "best_val_score": val_score,
        "best_test_score": test_score,
        "records": [asdict(record) for record in records],
    }
    run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_path = prompts_dir / "best_haiku_instruction.txt"
    best_path.write_text(best.instruction + "\n", encoding="utf-8")

    report_path = runs_dir / f"haiku_opro_report_{timestamp}.md"
    report_path.write_text(
        render_markdown_report(records, best, val_score, test_score, args),
        encoding="utf-8",
    )
    return run_path, report_path, best_path


def render_markdown_report(
    records: list[CandidateRecord],
    best: CandidateRecord,
    val_score: float,
    test_score: float,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# DSPy + OPRO Haiku Run Report",
        "",
        f"- Model: `{args.model}`",
        f"- Dataset: `{args.dataset}`",
        f"- Rounds: `{args.rounds}`",
        f"- Candidates per round: `{args.candidates_per_round}`",
        f"- Train / Val / Test size: `{args.train_size}` / `{args.val_size}` / `{args.test_size}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Best Instruction",
        "",
        "```text",
        best.instruction,
        "```",
        "",
        "## Scores",
        "",
        f"- Train: `{best.train_score:.4f}`",
        f"- Val: `{val_score:.4f}`",
        f"- Test: `{test_score:.4f}`",
        "",
        "## Candidate History",
        "",
    ]

    ranked = sorted(records, key=lambda rec: rec.train_score, reverse=True)
    for idx, record in enumerate(ranked, start=1):
        lines.extend(
            [
                f"### Candidate {idx}",
                "",
                f"- Source: `{record.source}`",
                f"- Train score: `{record.train_score:.4f}`",
                "",
                "Instruction:",
                "",
                "```text",
                record.instruction,
                "```",
                "",
                "Feedback:",
                "",
            ]
        )
        if record.feedback:
            lines.extend(f"- {item}" for item in record.feedback)
        else:
            lines.append("- No major failures.")
        lines.append("")

    return "\n".join(lines)


def score_on_split(instruction: str, examples: list[dspy.Example]) -> float:
    record = evaluate_instruction(instruction, examples, source="final_eval")
    return record.train_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=6)
    parser.add_argument("--val-size", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    api_key = load_api_key(args.api_key_file)
    lm = dspy.LM(args.model, api_key=api_key)
    dspy.configure(lm=lm)

    examples = load_examples(Path(args.dataset))
    train, val, test = split_examples(
        examples,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Using train={len(train)}, val={len(val)}, test={len(test)}")

    records: list[CandidateRecord] = []
    seen: set[str] = set()

    for idx, instruction in enumerate(seed_instructions(), start=1):
        print(f"\nEvaluating seed instruction {idx}...")
        record = evaluate_instruction(instruction, train, source=f"seed_{idx}")
        records.append(record)
        seen.add(instruction)
        print(f"score={record.train_score:.4f}")

    for round_idx in range(1, args.rounds + 1):
        print(f"\nOPRO round {round_idx}: proposing instructions...")
        proposed = propose_instructions(records, args.candidates_per_round)
        if not proposed:
            print("No parseable optimizer output; stopping early.")
            break

        for idx, instruction in enumerate(proposed, start=1):
            if instruction in seen:
                continue
            seen.add(instruction)
            print(f"Evaluating OPRO candidate {round_idx}.{idx}...")
            record = evaluate_instruction(
                instruction,
                train,
                source=f"opro_round_{round_idx}",
            )
            records.append(record)
            print(f"score={record.train_score:.4f}")

    best = max(records, key=lambda rec: rec.train_score)
    val_score = score_on_split(best.instruction, val)
    test_score = score_on_split(best.instruction, test)
    run_path, report_path, best_path = save_results(
        records,
        best,
        val_score,
        test_score,
        args,
    )

    print("\nBest instruction")
    print("----------------")
    print(best.instruction)
    print(
        f"\nScores: train={best.train_score:.4f}, "
        f"val={val_score:.4f}, test={test_score:.4f}"
    )
    print(f"Saved run: {run_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved best prompt: {best_path}")


if __name__ == "__main__":
    main()
