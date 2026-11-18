"""Durable Gemini file batches. Never resubmit a job whose outcome is unknown."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable
import uuid

from google import genai
from google.genai import types

from .gemini_transport import gemini_http_options
from .generation_failures import TerminalGenerationFailure, require_complete_gemini_answer


TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


class GeminiBatchPending(RuntimeError):
    """The remote job is durable and running; the scheduler can release its worker."""


def ensure_no_pending_batch(artifact: Path) -> None:
    reconcile_checkpointed_batches(artifact, require_finished=True)


def _saved_request_keys(artifact: Path, *, attempted_job: str | None = None) -> set[str]:
    keys = set()
    if not artifact.is_file():
        return keys
    with artifact.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("request_key"), str):
                continue
            if "error" not in row or (attempted_job is not None and row.get("batch_job") == attempted_job):
                keys.add(row["request_key"])
    return keys


def _unconsumed_keys(path: Path, manifest: dict, artifact: Path) -> set[str]:
    if "unconsumed_keys" in manifest:
        return set(manifest["unconsumed_keys"])
    if manifest["phase"] != "imported":
        return set(manifest["keys"])
    # Older importers marked the whole job imported even on a limited run.
    # Only cached answers and terminal outcomes could avoid a fresh request.
    cached = set()
    unseen = set(manifest["keys"]) - _saved_request_keys(artifact, attempted_job=manifest.get("job_name"))
    output = path.with_suffix(".responses.jsonl")
    if unseen and output.is_file():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("error") or row.get("key") not in unseen:
                    continue
                try:
                    response_text(row.get("response") or {})
                except TerminalGenerationFailure:
                    pass
                except (ValueError, KeyError, TypeError):
                    continue
                cached.add(row["key"])
    return cached


def reconcile_checkpointed_batches(artifact: Path, *, require_finished: bool = False) -> None:
    """Finish a downloaded job if every key already has a successful checkpoint."""
    directory = state_directory(artifact)
    if not directory.exists():
        return
    with _lock(directory):
        saved_keys = None
        for path in directory.glob("*.job.json"):
            manifest = json.loads(path.read_text())
            phase = manifest["phase"]
            if phase in ("rejected", "abandoned") or (phase == "imported" and manifest.get("unconsumed_keys") == []):
                continue
            if phase in ("imported", "partially_imported") or (
                phase == "submitted" and manifest.get("remote_state") in TERMINAL_STATES
                and path.with_suffix(".responses.jsonl").is_file()
            ):
                remaining = _unconsumed_keys(path, manifest, artifact)
                if remaining:
                    if saved_keys is None:
                        saved_keys = _saved_request_keys(artifact)
                    remaining.difference_update(saved_keys)
                if not remaining or phase == "imported":
                    manifest.update(phase="partially_imported" if remaining else "imported", unconsumed_keys=sorted(remaining))
                    _save(path, manifest)
                if not remaining:
                    continue
            if require_finished:
                raise RuntimeError(f"An unfinished Gemini batch exists at {path}; resume with --gemini-execution batch")


def request_key(model: str, request: dict) -> str:
    payload = json.dumps([model, request], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def state_directory(artifact: Path) -> Path:
    artifact = artifact.resolve()
    return artifact.parent / "gemini_batch_jobs" / artifact.name


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _save(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


@contextmanager
def _lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.write(b"\0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is handling this Gemini batch: {directory}") from exc
        yield


def _new_client():
    options = gemini_http_options()
    # Batch API is documented on v1beta; synchronous calls keep using v1.
    options.api_version = "v1beta"
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options=options)


def response_text(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no answer: {response.get('promptFeedback', {})}")
    candidate = candidates[0]
    reason = candidate.get("finishReason", candidate.get("finish_reason"))
    require_complete_gemini_answer(reason)
    content = candidate.get("content") or {}
    text = "".join(
        part.get("text", "") for part in (content.get("parts") or [])
        if not part.get("thought")
    )
    if not text.strip():
        raise ValueError("Gemini returned an empty answer")
    return text


def run_batch(
    *, model: str, requests: dict[str, dict], artifact: Path,
    consume: Callable[[str, dict, bool], bool],
    on_wait: Callable[[str], None] | None = None,
    poll_seconds: float = 30,
    retry_unconfirmed: bool = False,
    wait: bool = True,
    sync_fallback_max: int = 0,
    client=None,
) -> None:
    """Import cached/pending jobs, then submit at most one attempt per missing key.

    ``consume`` checkpoints a result and returns whether it is accounted for
    (a valid response or a recorded terminal generation failure). Its third
    argument identifies replay of an already imported job. A callback/disk error
    propagates; the downloaded response remains available for the next run.
    """
    if poll_seconds <= 0:
        raise ValueError("Batch poll interval must be positive")
    if type(sync_fallback_max) is not int or sync_fallback_max < 0:
        raise ValueError("Sync fallback maximum must be a non-negative integer")
    if not requests:
        reconcile_checkpointed_batches(artifact)
        return
    directory = state_directory(artifact)
    pending = set(requests)
    attempted: set[str] = set()
    failed: dict[str, dict] = {}
    fallback_attempted: set[str] = set()
    owned_client = client is None

    def remote():
        nonlocal client
        if client is None:
            client = _new_client()
        return client

    def retrieve(manifest_path: Path, manifest: dict) -> bool:
        interested = pending.intersection(manifest["keys"])
        if not interested or manifest["phase"] in ("rejected", "abandoned"):
            return True
        output_path = manifest_path.with_suffix(".responses.jsonl")
        if not output_path.exists():
            if manifest["phase"] == "submitting":
                matches = [job for job in remote().batches.list()
                           if job.display_name == manifest["display_name"]]
                if len(matches) > 1:
                    raise RuntimeError(f"Multiple jobs match {manifest['display_name']}; refusing duplicate work")
                if not matches:
                    if retry_unconfirmed:
                        manifest["phase"] = "abandoned"
                        _save(manifest_path, manifest)
                        return True
                    raise RuntimeError(
                        f"Submission outcome unknown for {manifest['display_name']}. "
                        f"State: {manifest_path}. Rerun to discover the remote job. "
                        "Only after confirming no job exists, use --gemini-batch-retry-unconfirmed. "
                        "No replacement batch was submitted."
                    )
                manifest.update(job_name=matches[0].name, phase="submitted")
                _save(manifest_path, manifest)
            if manifest["phase"] == "prepared":
                input_path = manifest_path.with_suffix(".requests.jsonl")
                if not manifest.get("input_file"):
                    uploaded = remote().files.upload(
                        file=input_path, config={"mime_type": "jsonl"},
                    )
                    if not uploaded.name:
                        raise RuntimeError("Gemini upload returned no file name")
                    manifest["input_file"] = uploaded.name
                    _save(manifest_path, manifest)
                # Commit intent before the non-idempotent create operation.
                manifest["phase"] = "submitting"
                _save(manifest_path, manifest)
                try:
                    job = remote().batches.create(
                        model=model, src=manifest["input_file"], config={
                            "display_name": manifest["display_name"],
                            "http_options": types.HttpOptions(
                                retry_options=types.HttpRetryOptions(attempts=1),
                            ),
                        },
                    )
                except Exception as exc:
                    if getattr(exc, "code", None) in (400, 401, 403, 404, 413, 422, 429):
                        manifest["phase"] = "rejected"
                        _save(manifest_path, manifest)
                    raise
                if not job.name:
                    raise RuntimeError("Gemini create returned no job name; submission needs reconciliation")
                manifest.update(job_name=job.name, phase="submitted")
                _save(manifest_path, manifest)
                print(f"[batch] Submitted {job.name}: {len(manifest['keys'])} requests; state: {manifest_path}")
            if not manifest.get("job_name"):
                raise RuntimeError(f"Invalid batch checkpoint: {manifest_path}")
            print(f"[batch] Retrieving remote job {manifest['job_name']} (local interruption does not cancel it)")
            previous_state = None
            while True:
                job = remote().batches.get(name=manifest["job_name"])
                state = getattr(job.state, "name", job.state)
                if state != previous_state:
                    print(f"[batch] {manifest['job_name']}: {state}")
                    previous_state = state
                if state in TERMINAL_STATES:
                    break
                if on_wait:
                    on_wait(str(state))
                if not wait:
                    raise GeminiBatchPending(f"{manifest['job_name']}: {state}")
                time.sleep(poll_seconds)
            if job.dest and job.dest.file_name:
                content = remote().files.download(file=job.dest.file_name)
                if not isinstance(content, bytes):
                    raise ValueError("Gemini returned no batch output file")
            elif state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
                raise ValueError(f"Completed batch has no output file: {manifest['job_name']}")
            else:
                content = b""  # Missing keys below receive the terminal job error.
            manifest["remote_state"] = state
            manifest["remote_error"] = str(job.error or "")
            _save(manifest_path, manifest)
            _atomic_bytes(output_path, content)

        rows = {}
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if key not in manifest["keys"] or key in rows:
                raise ValueError(f"Unexpected or duplicate batch response key: {key}")
            rows[key] = row
        unconsumed = _unconsumed_keys(manifest_path, manifest, artifact)
        for key in manifest["keys"]:
            if key not in interested:
                continue
            row = rows.get(key, {"error": {
                "message": f"No result for request; {manifest.get('remote_state')}: {manifest.get('remote_error', '')}",
            }})
            row = {**row, "batch_job": manifest.get("job_name")}
            replay = key not in unconsumed
            if consume(key, row, replay):
                pending.discard(key)
            else:
                failed[key] = row
            if not replay:
                attempted.add(key)
        remaining = unconsumed - interested
        if remaining:
            remaining.difference_update(_saved_request_keys(artifact))
        manifest.update(phase="partially_imported" if remaining else "imported", unconsumed_keys=sorted(remaining))
        _save(manifest_path, manifest)
        return True

    def recover_small_failure_set() -> bool:
        """Return false when the failure cutoff forbids further requests."""
        keys = [key for key in requests if key in pending and key in failed
                and key not in fallback_attempted]
        if not keys or len(keys) + len(fallback_attempted) > sync_fallback_max:
            return True
        print(f"[batch fallback] Retrying {len(keys)} failed requests synchronously at standard pricing; saved successes are reused")
        # Do not switch back to a new paid batch after a failed sync retry.
        attempted.update(keys)
        fallback_attempted.update(keys)
        consecutive_errors = 0
        for key in keys:
            request = requests[key]
            row = {"batch_job": failed[key].get("batch_job"), "execution_mode": "sync_fallback"}
            permanent_error = False
            try:
                response = remote().models.generate_content(
                    model=model, contents=request["contents"],
                    config=types.GenerateContentConfig.model_validate({
                        **request.get("generationConfig", {}),
                        "systemInstruction": request.get("systemInstruction"),
                    }),
                )
                row["response"] = response.model_dump(
                    mode="json", by_alias=True, exclude_none=True, exclude={"sdk_http_response"},
                )
            except Exception as exc:
                row["error"] = {"message": str(exc)}
                permanent_error = isinstance(exc, (TypeError, AttributeError)) or getattr(exc, "code", None) in (400, 401, 402, 403, 404, 422)
            # Cache the raw reply before importing it. A failed checkpoint can
            # then recover this response without paying for another retry.
            path = directory / f"{time.time_ns()}-{uuid.uuid4().hex}.sync.json"
            _save(path, {"model": model, "key": key, "row": row})
            if consume(key, row, False):
                pending.discard(key)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
            if permanent_error or consecutive_errors >= 3:
                print("[batch fallback] Stopped after request failures; saved successes preserved for resume")
                return False
        return True

    try:
        with _lock(directory):
            for path in sorted(directory.glob("*.sync.json")):
                cached = json.loads(path.read_text())
                if cached.get("model") == model and cached.get("key") in pending:
                    if consume(cached["key"], cached["row"], True):
                        pending.discard(cached["key"])
            for path in sorted(directory.glob("*.job.json")):
                manifest = json.loads(path.read_text())
                if manifest.get("model") == model:
                    retrieve(path, manifest)
            if not recover_small_failure_set():
                return
            remaining = [key for key in requests if key in pending - attempted]
            if remaining:
                identifier = uuid.uuid4().hex
                path = directory / f"{time.time_ns()}-{identifier}.job.json"
                content = "".join(json.dumps({"key": key, "request": requests[key]}, ensure_ascii=False) + "\n" for key in remaining).encode()
                if len(content) > 2_000_000_000:
                    raise ValueError("Gemini batch input exceeds the 2GB file limit")
                _atomic_bytes(path.with_suffix(".requests.jsonl"), content)
                manifest = {"version": 2, "model": model, "keys": remaining,
                            "display_name": f"cp-evals-{identifier}", "phase": "prepared"}
                _save(path, manifest)
                retrieve(path, manifest)
                recover_small_failure_set()
    finally:
        if owned_client and client is not None:
            client.close()
