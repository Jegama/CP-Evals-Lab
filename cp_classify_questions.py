"""Classify evaluation questions with LLM-based tagging for selective scoring.

Reads an evaluation questions file, classifies each question using an LLM with
structured output, and writes a JSON tags file used by cp_eval_llms.py to null
out inapplicable subcriteria during aggregation.

Usage:
    python cp_classify_questions.py
    python cp_classify_questions.py --model gpt-5-mini --questions-file data/english/en_eval_questions.txt
    python cp_classify_questions.py --resume   # Continue after API failure
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import BaseModel

from parrot_ai.evaluation_schemas import (
    DoctrineTier,
    QuestionTag,
    QuestionTagSet,
    QuestionType,
)
from parrot_ai.prompts.english import DOCTRINE_TIER_DEFINITIONS

load_dotenv()  # Load environment variables from .env file for OpenAI API key

CLASSIFICATION_SYSTEM_PROMPT = f"""You are a theology and question classification expert. Your task is to classify a theological question along two dimensions and set six boolean applicability flags.

## Doctrine Tier Definitions
Use the following detailed definitions to determine the doctrine tier.
Map 'Core Doctrines' to **core**, 'Secondary Doctrines' to **secondary**, and 'Tertiary Doctrines' to **tertiary**.

{DOCTRINE_TIER_DEFINITIONS}

Additional Tier:
- **not_directly_doctrinal**: Factual, historical, biographical, methodological, or practical questions that don't directly test doctrinal positions.

## Question Type Definitions
- **factual_historical**: Asks for facts, dates, people, events, or definitions (e.g., "Who was Herod Agrippa I?", "What is the Dead Sea Scrolls?")
- **doctrinal**: Directly asks about a doctrine or theological concept (e.g., "What is justification by faith?", "Explain the Trinity")
- **apologetic_interfaith**: Involves defending Christianity, comparing with other religions, or addressing skeptics (e.g., "How do Christians respond to the problem of evil?", "What do Christians say to Muslims about Jesus?")
- **pastoral**: Addresses suffering, grief, spiritual struggle, or practical spiritual life (e.g., "How should I deal with doubt?", "What does the Bible say about suffering?")
- **practical_ethical**: Asks about Christian ethics, moral decisions, or practical living (e.g., "Is it wrong to lie?", "What does the Bible say about divorce?")
- **comparative_religion**: Specifically compares Christian beliefs with other religions or worldviews without an apologetic framing
- **methodological**: Asks about methods of Bible study, hermeneutics, or theological methodology
- **bible_survey**: Asks about books, themes, or structure of the Bible (e.g., "What is the book of Romans about?", "What are the major themes of Genesis?")

## Flag Guidance
Set each flag to true ONLY if a good answer to this question SHOULD demonstrate the corresponding behavior. A flag being false means a good answer would NOT naturally exhibit that behavior for this particular question.

- **applies_core_doctrine**: True if the question directly asks about, requires explanation of, or necessarily involves a core doctrine. False for questions about secondary/tertiary matters, historical facts, or practical topics that don't require core doctrinal engagement.
- **applies_secondary_doctrine**: True if the question asks about a secondary doctrine or requires taking/acknowledging denominational positions. False for core-only, tertiary, factual, or non-doctrinal questions.
- **applies_tertiary_handling**: True if the question asks about a tertiary/disputable matter where neutrality and Christian liberty should be modeled. False for questions about core/secondary doctrine or non-doctrinal topics.
- **applies_pastoral**: True if the question involves emotional/spiritual struggle, suffering, doubt, grief, or personal spiritual life where pastoral sensitivity matters. False for purely factual, academic, or doctrinal-definition questions.
- **applies_interfaith**: True if the question involves other religions, skeptical objections, worldview comparison, or topics where a non-Christian perspective is relevant. False for intra-Christian questions, factual questions, or topics with no meaningful interfaith dimension.
- **applies_evangelism**: True if the question touches salvation, the identity of Christ, the meaning of life, guilt, forgiveness, eternity, or becoming a Christian, where a Gospel invitation would be natural. False for factual, methodological, intra-Christian doctrinal, or purely academic questions.

## Output
Classify the question and provide a brief reason (1-2 sentences) explaining your classification choices."""

CLASSIFICATION_USER_TEMPLATE = "Classify this question:\n\n{question}"


class ClassificationResult(BaseModel):
    """Single question classification for structured output."""

    doctrine_tier: DoctrineTier
    question_type: QuestionType
    applies_core_doctrine: bool
    applies_secondary_doctrine: bool
    applies_tertiary_handling: bool
    applies_pastoral: bool
    applies_interfaith: bool
    applies_evangelism: bool
    reason: str


def load_questions(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    return questions


def load_existing_tags(path: Path) -> dict[str, QuestionTag]:
    """Load existing tags file for resume support."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tag_set = QuestionTagSet(**data)
    return {tag.question: tag for tag in tag_set.tags}


def classify_question(
    client: OpenAI, model: str, question: str
) -> ClassificationResult:
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CLASSIFICATION_USER_TEMPLATE.format(question=question),
            },
        ],
        response_format=ClassificationResult,
        seed=42,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Failed to parse classification for: {question[:80]}...")
    return parsed


def print_summary(tags: List[QuestionTag]) -> None:
    print(f"\n{'='*60}")
    print(f"Classification Summary ({len(tags)} questions)")
    print(f"{'='*60}")

    # Doctrine tier distribution
    tier_counts: dict[str, int] = {}
    for tag in tags:
        tier_counts[tag.doctrine_tier.value] = (
            tier_counts.get(tag.doctrine_tier.value, 0) + 1
        )
    print("\nDoctrine Tier Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} ({100*count/len(tags):.1f}%)")

    # Question type distribution
    type_counts: dict[str, int] = {}
    for tag in tags:
        type_counts[tag.question_type.value] = (
            type_counts.get(tag.question_type.value, 0) + 1
        )
    print("\nQuestion Type Distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count} ({100*count/len(tags):.1f}%)")

    # Flag counts
    flag_names = [
        "applies_core_doctrine",
        "applies_secondary_doctrine",
        "applies_tertiary_handling",
        "applies_pastoral",
        "applies_interfaith",
        "applies_evangelism",
    ]
    print("\nFlag Counts (true):")
    for flag in flag_names:
        count = sum(1 for tag in tags if getattr(tag, flag))
        print(f"  {flag}: {count} ({100*count/len(tags):.1f}%)")

    # Questions with ALL flags false (purely factual)
    all_false = sum(
        1
        for tag in tags
        if not any(getattr(tag, f) for f in flag_names)
    )
    print(f"\nQuestions with all flags false: {all_false} ({100*all_false/len(tags):.1f}%)")


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Classify evaluation questions for selective scoring."
    )
    p.add_argument(
        "--questions-file",
        default="data/english/en_eval_questions.txt",
        help="Path to evaluation questions file (default: data/english/en_eval_questions.txt)",
    )
    p.add_argument(
        "--output",
        default="data/english/en_question_tags.json",
        help="Output JSON tags file (default: data/english/en_question_tags.json)",
    )
    p.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Classification model (default: gpt-5-mini)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial tags file",
    )
    args = p.parse_args(argv)

    questions = load_questions(args.questions_file)
    if not questions:
        print(f"[error] No questions found in {args.questions_file}")
        return 1
    print(f"[init] Loaded {len(questions)} questions from {args.questions_file}")
    print(f"[init] Classification model: {args.model}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing_tags: dict[str, QuestionTag] = {}
    if args.resume:
        existing_tags = load_existing_tags(output_path)
        if existing_tags:
            print(f"[resume] Found {len(existing_tags)} existing classifications")

    client = OpenAI()
    all_tags: List[QuestionTag] = []
    errors = 0

    for i, question in enumerate(questions, 1):
        # Skip already classified questions on resume
        if question in existing_tags:
            all_tags.append(existing_tags[question])
            continue

        try:
            result = classify_question(client, args.model, question)
            tag = QuestionTag(
                question=question,
                doctrine_tier=result.doctrine_tier,
                question_type=result.question_type,
                applies_core_doctrine=result.applies_core_doctrine,
                applies_secondary_doctrine=result.applies_secondary_doctrine,
                applies_tertiary_handling=result.applies_tertiary_handling,
                applies_pastoral=result.applies_pastoral,
                applies_interfaith=result.applies_interfaith,
                applies_evangelism=result.applies_evangelism,
                reason=result.reason,
            )
            all_tags.append(tag)
            print(f"  [{i}/{len(questions)}] {question[:60]}... -> {result.question_type.value} ({result.doctrine_tier.value})")
        except Exception as e:
            print(f"  [{i}/{len(questions)}] ERROR: {question[:60]}... -> {e}")
            errors += 1
            # Save progress on error so --resume can continue
            if all_tags:
                _save_tags(output_path, all_tags, args.model)
                print(f"  [save] Progress saved ({len(all_tags)} tags). Re-run with --resume to continue.")
            return 1

    # Save final output
    _save_tags(output_path, all_tags, args.model)
    print(f"\n[done] Wrote {len(all_tags)} tags to {output_path}")
    if errors:
        print(f"[warn] {errors} errors encountered")

    print_summary(all_tags)
    return 0


def _save_tags(path: Path, tags: List[QuestionTag], model: str) -> None:
    tag_set = QuestionTagSet(
        tags=tags,
        classification_model=model,
        classification_timestamp=dt.now().isoformat(),
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(json.loads(tag_set.model_dump_json()), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
