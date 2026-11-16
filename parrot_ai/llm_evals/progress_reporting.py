"""Lightweight progress events for subprocess benchmark orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROGRESS_FILE_ENV = "CP_EVAL_PROGRESS_FILE"


def emit_progress(stage: str, current: int, total: int | None, *, reused: int | None = None) -> None:
    """Append one progress event when a parent benchmark runner requested it."""

    progress_file = os.environ.get(PROGRESS_FILE_ENV)
    if not progress_file:
        return

    event = {
        "stage": stage,
        "current": current,
        "total": total,
    }
    if reused is not None:
        event["reused"] = reused
    try:
        with Path(progress_file).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    except OSError:
        # Progress display must never turn a successful provider call into a failure.
        return
