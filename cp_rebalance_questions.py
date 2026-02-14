"""Rebalance the English evaluation question set using gotquestions categories.

Detects distribution deficits in en_question_tags.json, pulls targeted candidates
from gotquestions categories mapped to each deficit, classifies them via LLM, and
swaps them in one-at-a-time (removing the lowest-value question each time) until
all distribution goals are met. Total stays at 500.

Usage:
    python cp_rebalance_questions.py              # Run rebalancing
    python cp_rebalance_questions.py --dry-run    # Preview without writing

Prerequisites:
    Existing `data/english/en_question_tags.json` from cp_classify_questions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI

from cp_classify_questions import (
    classify_question,
    print_summary,
)
from parrot_ai.evaluation_schemas import QuestionTag, QuestionTagSet

load_dotenv()

# Distribution goals
GOAL_TOTAL = 500
GOAL_TIER_MIN = {
    "core": 100,
    "secondary": 100,
    "tertiary": 100,
}
GOAL_FLAG_MIN = {
    "applies_interfaith": 100,
    "applies_evangelism": 100,
}

# GotQuestions categories mapped to the deficit they're most likely to fill.
# Each deficit key maps to a list of (category_name, theme_filter_or_None).
DEFICIT_CATEGORY_MAP: dict[str, list[tuple[str, Optional[list[str]]]]] = {
    "flag:applies_interfaith": [
        ("Questions about Apologetics", None),
        ("Questions about Cults and Religions", None),
        ("Questions about Islam", None),
        ("Questions about Catholicism", None),
        ("Questions about False Beliefs", None),
        ("Questions about Judaism", None),
        ("Questions about Worldview", None),
    ],
    "flag:applies_evangelism": [
        ("Questions about Salvation", None),
        ("Questions about Eternity", None),
    ],
    "tier:core": [
        ("Questions about God", None),
        ("Questions about Jesus Christ", None),
        ("Questions about the Holy Spirit", None),
        ("Questions about Theology", None),
    ],
    "tier:secondary": [
        ("Questions about the Church", None),
        ("Questions about Theology", None),
    ],
    "tier:tertiary": [
        ("Questions about the End Times", None),
        ("Questions about Family", None),
        ("Questions about Life", None),
        ("Questions about Relationships", None),
    ],
}

FLAG_NAMES = [
    "applies_core_doctrine",
    "applies_secondary_doctrine",
    "applies_tertiary_handling",
    "applies_pastoral",
    "applies_interfaith",
    "applies_evangelism",
]


def load_gotquestions(path: Path) -> Dict[str, dict]:
    """Load gotquestions data, return dict keyed by question name."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions: Dict[str, dict] = {}
    for cat in data:
        cat_name = cat["name"]
        for theme in cat.get("themes", []):
            theme_name = theme["name"]
            articles = theme.get("articles", {})
            items = []
            if isinstance(articles, dict):
                for group_items in articles.values():
                    items.extend(group_items)
            elif isinstance(articles, list):
                items = articles
            for item in items:
                if item.get("answer", "").strip():
                    questions[item["name"]] = {
                        "category": cat_name,
                        "theme": theme_name,
                    }
    return questions


def extract_candidates(
    gotq: Dict[str, dict],
    existing: Set[str],
    categories: list[tuple[str, Optional[list[str]]]],
) -> List[str]:
    """Extract candidate questions from gotquestions matching the category config."""
    candidates = []
    for cat_name, theme_filter in categories:
        for q, meta in gotq.items():
            if q in existing:
                continue
            if meta["category"] != cat_name:
                continue
            if theme_filter is not None and meta["theme"] not in theme_filter:
                continue
            candidates.append(q)
    # Deduplicate preserving order
    seen: Set[str] = set()
    deduped = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def compute_deficits(tags: List[QuestionTag]) -> Dict[str, int]:
    """Compute how many more questions are needed for each goal."""
    from collections import Counter

    tier_counts = Counter(t.doctrine_tier.value for t in tags)
    flag_counts = {f: sum(1 for t in tags if getattr(t, f)) for f in FLAG_NAMES}

    deficits = {}
    deficits["total"] = max(0, GOAL_TOTAL - len(tags))
    for tier, goal in GOAL_TIER_MIN.items():
        deficits[f"tier:{tier}"] = max(0, goal - tier_counts.get(tier, 0))
    for flag, goal in GOAL_FLAG_MIN.items():
        deficits[f"flag:{flag}"] = max(0, goal - flag_counts.get(flag, 0))
    return deficits


def question_helps_deficit(tag: QuestionTag, deficits: Dict[str, int]) -> bool:
    """Check if a classified question helps fill any remaining deficit."""
    if deficits.get("total", 0) > 0:
        return True  # Any question helps when we're under GOAL_TOTAL
    tier_key = f"tier:{tag.doctrine_tier.value}"
    if deficits.get(tier_key, 0) > 0:
        return True
    for flag in FLAG_NAMES:
        flag_key = f"flag:{flag}"
        if deficits.get(flag_key, 0) > 0 and getattr(tag, flag):
            return True
    return False


def find_removable(tags: List[QuestionTag], count: int) -> List[QuestionTag]:
    """Rank questions by removability (most removable first).

    Priority: all-flags-false questions from not_directly_doctrinal tier,
    preferring factual_historical type.
    """
    scored = []
    for tag in tags:
        flag_count = sum(1 for f in FLAG_NAMES if getattr(tag, f))
        priority = 0
        if tag.doctrine_tier.value == "not_directly_doctrinal":
            priority += 100
        if tag.question_type.value == "factual_historical":
            priority += 50
        if tag.question_type.value == "bible_survey":
            priority += 40
        if tag.question_type.value == "methodological":
            priority += 30
        # Fewer flags = more removable
        priority += (6 - flag_count) * 10
        scored.append((priority, tag))

    scored.sort(key=lambda x: -x[0])
    return [tag for _, tag in scored[:count]]


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Rebalance evaluation questions.")
    p.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Classification model (default: gpt-5-mini)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    args = p.parse_args(argv)

    base_dir = Path("data/english")
    tags_path = base_dir / "en_question_tags.json"
    questions_path = base_dir / "en_eval_questions.txt"
    gotq_path = base_dir / "en_gotquestions.json"

    # Load current tags
    with tags_path.open("r", encoding="utf-8") as f:
        tags_data = json.load(f)
    current_tags = [QuestionTag(**t) for t in tags_data["tags"]]
    print(f"[init] Current tags: {len(current_tags)}")

    # Check deficits
    deficits = compute_deficits(current_tags)
    total_deficit = sum(deficits.values())
    if total_deficit == 0:
        print("[done] All goals already met, nothing to do.")
        return 0

    print(f"\n[deficits] Current:")
    for k, v in deficits.items():
        if v > 0:
            print(f"  {k}: need {v} more")

    # Load gotquestions
    gotq = load_gotquestions(gotq_path)
    existing_questions = {t.question for t in current_tags}

    # Build per-deficit candidate queues (only for deficits that exist)
    candidate_queues: dict[str, deque[str]] = {}
    for deficit_key, categories in DEFICIT_CATEGORY_MAP.items():
        if deficits.get(deficit_key, 0) > 0:
            candidates = extract_candidates(gotq, existing_questions, categories)
            candidate_queues[deficit_key] = deque(candidates)
            print(f"  {deficit_key}: {len(candidates)} candidates available")

    # If only a total deficit remains (no tier/flag deficits), pull from all categories
    if deficits.get("total", 0) > 0 and not candidate_queues:
        all_categories = []
        for cats in DEFICIT_CATEGORY_MAP.values():
            all_categories.extend(cats)
        candidates = extract_candidates(gotq, existing_questions, all_categories)
        candidate_queues["total"] = deque(candidates)
        print(f"  total: {len(candidates)} candidates available (filling to {GOAL_TOTAL})")

    # Pre-rank all current questions by removability (most removable first)
    removable_ranked = find_removable(current_tags, len(current_tags))

    # Add/swap one-at-a-time, deficit-driven
    client = OpenAI()
    all_tags = list(current_tags)
    current_questions = {t.question for t in all_tags}
    classified: Set[str] = set()
    added = 0
    swapped = 0
    skipped = 0
    errors = 0

    while True:
        current_deficits = compute_deficits(all_tags)
        if all(v == 0 for v in current_deficits.values()):
            print(f"\n[done] All goals met after {added} additions and {swapped} swaps!")
            break

        need_to_fill = current_deficits.get("total", 0) > 0

        # Pick the deficit with the highest remaining count that has candidates
        # Prioritize tier/flag deficits over total deficit
        best_key = None
        best_count = 0
        for dk, count in current_deficits.items():
            if dk == "total":
                continue  # Check total last
            if count > best_count and dk in candidate_queues and candidate_queues[dk]:
                best_key = dk
                best_count = count

        # Fall back to total deficit if no tier/flag deficits have candidates
        if best_key is None and need_to_fill:
            if "total" in candidate_queues and candidate_queues["total"]:
                best_key = "total"
                best_count = current_deficits["total"]
            else:
                # Try any queue that still has candidates
                for dk in candidate_queues:
                    if candidate_queues[dk]:
                        best_key = dk
                        break

        if best_key is None:
            print(f"\n[exhausted] No more candidates for remaining deficits")
            break

        # Pop next unclassified candidate from that deficit's queue
        question = None
        while candidate_queues[best_key]:
            q = candidate_queues[best_key].popleft()
            if q not in classified and q not in current_questions:
                question = q
                break

        if question is None:
            # This queue is exhausted, remove it and retry
            del candidate_queues[best_key]
            continue

        classified.add(question)

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

            # Only accept if this question helps fill a remaining deficit
            if question_helps_deficit(tag, current_deficits):
                # Show what deficit it helped
                helped = []
                tier_key = f"tier:{tag.doctrine_tier.value}"
                if current_deficits.get(tier_key, 0) > 0:
                    helped.append(tier_key)
                for flag in FLAG_NAMES:
                    flag_key = f"flag:{flag}"
                    if current_deficits.get(flag_key, 0) > 0 and getattr(tag, flag):
                        helped.append(flag_key)

                if need_to_fill:
                    # Under GOAL_TOTAL: just add, no swap needed
                    all_tags.append(tag)
                    current_questions.add(question)
                    added += 1
                    if current_deficits["total"] > 0:
                        helped.append("total")
                    print(
                        f"  [{added + swapped:3d}] ADD [{tag.doctrine_tier.value}] "
                        f"{question[:60]}..."
                        f"\n        helps: {', '.join(helped)}"
                        f"  (targeting {best_key})"
                    )
                else:
                    # At GOAL_TOTAL: swap (remove least valuable, add new)
                    victim = None
                    for candidate_victim in removable_ranked:
                        if candidate_victim.question in current_questions:
                            victim = candidate_victim
                            break
                    if victim is None:
                        print(f"  [warn] No removable questions left, stopping")
                        break

                    all_tags = [t for t in all_tags if t.question != victim.question]
                    current_questions.discard(victim.question)
                    all_tags.append(tag)
                    current_questions.add(question)
                    swapped += 1
                    print(
                        f"  [{added + swapped:3d}] SWAP IN [{tag.doctrine_tier.value}] "
                        f"{question[:60]}..."
                        f"\n        OUT: {victim.question[:50]}..."
                        f"\n        helps: {', '.join(helped)}"
                        f"  (targeting {best_key})"
                    )
            else:
                skipped += 1

        except Exception as e:
            print(f"  ERROR: {question[:60]}... -> {e}")
            errors += 1
            if errors >= 5:
                print("[error] Too many errors, saving progress")
                break

    print(f"\n[summary] Added {added}, swapped {swapped}, skipped {skipped}, errors {errors}")

    # Final distribution check
    final_deficits = compute_deficits(all_tags)
    print(f"\n[final] Distribution after rebalancing ({len(all_tags)}/{GOAL_TOTAL} questions):")
    all_met = True
    for k, v in final_deficits.items():
        status = "OK" if v == 0 else f"STILL NEED {v}"
        if v > 0:
            all_met = False
        print(f"  {k}: {status}")

    if not all_met:
        print("\n[warn] Not all goals met. May need more categories in DEFICIT_CATEGORY_MAP.")

    print_summary(all_tags)

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    # Write updated tags
    tag_set = QuestionTagSet(
        tags=all_tags,
        classification_model=args.model,
        classification_timestamp=dt.now().isoformat(),
    )
    with tags_path.open("w", encoding="utf-8") as f:
        json.dump(
            json.loads(tag_set.model_dump_json()),
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[write] Updated tags -> {tags_path} ({len(all_tags)} questions)")

    # Write updated eval questions file
    with questions_path.open("w", encoding="utf-8") as f:
        for tag in all_tags:
            f.write(tag.question + "\n")
    print(f"[write] Updated eval questions -> {questions_path} ({len(all_tags)} questions)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
