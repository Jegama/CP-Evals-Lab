"""Recover a torn final checkpoint without rewriting completed JSONL records."""

import json
import os
from pathlib import Path
import tempfile


def recover_jsonl_tail(path: Path) -> None:
    """Quarantine only an invalid, unterminated final line before reuse/append.

    Completed lines and valid JSON without a final newline are left byte-for-byte
    intact. Callers must own the artifact exclusively, as with checkpoint writes.
    Corruption in newline-terminated records remains a loader error.
    """
    if not path.exists():
        return
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if not end:
            return
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return

        # Find the final line in blocks; avoid rereading the whole artifact on
        # every checkpoint, including files with large multilingual answers.
        start = end
        while start:
            block_start = max(0, start - 8192)
            stream.seek(block_start)
            block = stream.read(start - block_start)
            newline = block.rfind(b"\n")
            if newline >= 0:
                start = block_start + newline + 1
                break
            start = block_start
        stream.seek(start)
        tail = stream.read(end - start)
        if not tail.strip():
            return
        try:
            json.loads(tail.decode("utf-8"))
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # Preserve the fragment durably before touching the source. If disk
    # space is still exhausted, this raises and leaves the source intact.
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=path.name + ".incomplete-", dir=path.parent, delete=False,
    ) as backup:
        backup.write(tail)
        backup.flush()
        os.fsync(backup.fileno())
        backup_path = backup.name
    with path.open("r+b") as stream:
        stream.truncate(start)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"[recovery] Quarantined incomplete final record from {path} at {backup_path}")
