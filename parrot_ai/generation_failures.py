"""Explicit terminal answer-generation outcomes, separate from API failures."""

import json
from pathlib import Path


TERMINAL_GENERATION_REASONS = frozenset({"RECITATION"})


class TerminalGenerationFailure(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Gemini answer did not finish successfully: {reason}")

    def metadata(self) -> dict:
        return {"reason": self.reason, "terminal": True}


def is_terminal_generation_failure(record: object) -> bool:
    if not isinstance(record, dict) or "error" in record:
        return False
    failure = record.get("generation_failure")
    return (
        isinstance(failure, dict)
        and failure.get("terminal") is True
        and isinstance(failure.get("reason"), str)
        and failure.get("reason") in TERMINAL_GENERATION_REASONS
    )


def load_generation_failures(path: Path) -> dict[str, dict]:
    """Read explicit failed outcomes; ordinary blanks/errors remain unfinished."""
    failures = {}
    if not path.exists():
        return failures
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not is_terminal_generation_failure(row):
            continue
        messages = row.get("messages") or []
        questions = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        answers = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        if (len(questions) == len(answers) == 1 and isinstance(questions[0], str)
                and questions[0].strip() and isinstance(answers[0], str) and not answers[0].strip()):
            failures[questions[0]] = row
    return failures
