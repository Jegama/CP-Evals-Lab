"""Data loading utilities for evaluation QA pairs and question lists."""

import json
from typing import List, Tuple, Optional


def load_qa_pairs(
    jsonl_path: str,
    question_list_path: Optional[str] = "data/arabic/ar_eval_questions.txt",
    limit: int = 100,
) -> List[Tuple[str, str]]:
    """Load (question, answer) pairs filtered to a curated eval question list.

    Behavior change (per user request): Instead of loading every pair in the
    dataset, we now restrict to the first ``limit`` questions present in the
    ``question_list_path`` file (one question per line) if that file exists.

    If ``question_list_path`` is None or the file is missing/empty, the
    function falls back to previous behavior (load all pairs) but still caps
    at ``limit`` if provided (>0).
    """
    question_filter: Optional[List[str]] = None
    question_set: Optional[set[str]] = None
    if question_list_path and limit != 0:
        try:
            with open(question_list_path, "r", encoding="utf-8") as qf:
                question_filter = [ln.strip() for ln in qf if ln.strip()][:limit]
            if question_filter:
                question_set = set(question_filter)
        except FileNotFoundError:
            question_filter = None
            question_set = None
    pairs: List[Tuple[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if question_set and len(pairs) >= len(question_set):
                break
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = obj.get("messages") or []
            if len(messages) >= 3:
                user = (
                    messages[1].get("content")
                    if isinstance(messages[1], dict)
                    else None
                )
                assistant = (
                    messages[2].get("content")
                    if isinstance(messages[2], dict)
                    else None
                )
                if user and assistant:
                    if question_set is None or user in question_set:
                        pairs.append((user, assistant))
            elif len(messages) == 2:
                user = (
                    messages[0].get("content")
                    if isinstance(messages[0], dict)
                    else None
                )
                assistant = (
                    messages[1].get("content")
                    if isinstance(messages[1], dict)
                    else None
                )
                if user and assistant:
                    if question_set is None or user in question_set:
                        pairs.append((user, assistant))
    # Re-order according to the question list if present
    if question_filter and pairs:
        order_map = {q: i for i, q in enumerate(question_filter)}
        pairs.sort(key=lambda qa: order_map.get(qa[0], 10_000_000))
    # Apply limit fallback if no question list was used
    if question_set is None and limit and limit > 0:
        pairs = pairs[:limit]
    return pairs


def load_eval_questions(
    question_file: str = "data/arabic/ar_eval_questions.txt", limit: Optional[int] = 100
) -> List[str]:
    """Load up to ``limit`` evaluation questions from text file (one per line).

    If limit is None or 0, load all questions.
    """
    try:
        with open(question_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            if limit and limit > 0:
                return lines[:limit]
            return lines
    except FileNotFoundError:
        return []
