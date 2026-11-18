"""Exercise torn writes and conservative recovery without provider calls."""

import json
from pathlib import Path

import pytest

from cp_eval_llms import _append_jsonl_record
from parrot_ai import jsonl_checkpoints as checkpoints
from parrot_ai.llm_evals.master_csv import load_jsonl


def test_disk_full_mid_write_recovers_without_losing_completed_records(tmp_path, monkeypatch):
    path = tmp_path / "reviews.jsonl"
    _append_jsonl_record(path, {"question": "q1", "answer": "Saved answer"})
    completed = path.read_bytes()
    original_open = Path.open

    class DiskFullWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, payload):
            self.stream.write(payload[:len(payload) // 2])
            self.stream.flush()
            raise OSError("disk full")

    def failing_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return DiskFullWrite(stream) if mode == "a+b" else stream

    record = {"question": "q2", "answer": "Retry this checkpoint"}
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="disk full"):
            _append_jsonl_record(path, record)
    fragment = path.read_bytes()[len(completed):]
    assert fragment and not fragment.endswith(b"\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(path)

    _append_jsonl_record(path, record)
    assert path.read_bytes().startswith(completed)
    assert [row["question"] for row in load_jsonl(path)] == ["q1", "q2"]
    backups = list(tmp_path.glob(path.name + ".incomplete-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == fragment
    recovered = path.read_bytes()
    checkpoints.recover_jsonl_tail(path)
    assert path.read_bytes() == recovered
    assert list(tmp_path.glob(path.name + ".incomplete-*")) == backups


def test_quarantine_failure_leaves_original_artifact_untouched(tmp_path, monkeypatch):
    path = tmp_path / "answers.jsonl"
    original = b'{"question": "q1"}\n{"question": "q2'
    path.write_bytes(original)

    def disk_full(*args):
        raise OSError("disk full while saving quarantine")

    monkeypatch.setattr(checkpoints.os, "fsync", disk_full)
    with pytest.raises(OSError, match="disk full"):
        checkpoints.recover_jsonl_tail(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("suffix", [b"", b'{"question": "q3"}'])
def test_recovery_does_not_hide_corruption_in_completed_lines(tmp_path, suffix):
    path = tmp_path / "answers.jsonl"
    original = b'{"question": "q1"}\n{"question":\n' + suffix
    path.write_bytes(original)
    checkpoints.recover_jsonl_tail(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(path.name + ".incomplete-*"))
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(path)


def test_recovery_handles_a_fragment_larger_than_a_read_block(tmp_path):
    path = tmp_path / "answers.jsonl"
    completed = (json.dumps({"question": "q1", "answer": "أ" * 10000}, ensure_ascii=False) + "\n").encode()
    fragment = b'{"answer": "' + b"x" * 20000
    path.write_bytes(completed + fragment)
    checkpoints.recover_jsonl_tail(path)
    assert path.read_bytes() == completed
    assert next(tmp_path.glob(path.name + ".incomplete-*")).read_bytes() == fragment
