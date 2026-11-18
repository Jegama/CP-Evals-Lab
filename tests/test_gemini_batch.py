"""Offline tests for paid-job identity, restart safety, and batch integration."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types
import pytest

import cp_eval_llms as cli
from parrot_ai import gemini_batch as batch
from parrot_ai.evaluation_schemas import EvaluationResultEnglish
from parrot_ai.llm_evaluation import EvaluationEngine
from parrot_ai.llm_evals.gemini_batch_adapter import generate_gemini_batch, judge_gemini_batch


def response(text="answer"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 4, "thoughtsTokenCount": 7}}


class FakeClient:
    def __init__(self, cloud=None):
        self.cloud = cloud
        self.uploads = {}
        self.created = []
        if cloud and cloud.exists():
            saved = json.loads(cloud.read_text())
            self.uploads, self.created = saved["uploads"], saved["created"]
        self.rows = lambda request: {"response": response(request["contents"][0]["parts"][0]["text"])}
        self.files = SimpleNamespace(upload=Mock(side_effect=self.upload), download=Mock(side_effect=self.download))
        self.batches = SimpleNamespace(create=Mock(side_effect=self.create), get=Mock(side_effect=self.get),
                                       list=Mock(side_effect=lambda: [self.job(i) for i in range(len(self.created))]))
        self.close = Mock()
        self.models = SimpleNamespace(generate_content=Mock(side_effect=self.generate_content))

    def generate_content(self, *, model, contents, config):
        return types.GenerateContentResponse.model_validate(response("recovered " + contents[0]["parts"][0]["text"]))

    def upload(self, *, file, config):
        assert config["mime_type"] == "jsonl"
        name = f"files/input-{len(self.uploads)}"
        self.uploads[name] = [json.loads(line) for line in Path(file).read_text().splitlines()]
        return types.File(name=name)

    def create(self, *, model, src, config):
        assert config["http_options"].retry_options.attempts == 1
        self.created.append({"model": model, "src": src, "display_name": config["display_name"]})
        if self.cloud:
            self.cloud.write_text(json.dumps({"uploads": self.uploads, "created": self.created}))
        return self.job(len(self.created) - 1)

    def job(self, index):
        return types.BatchJob(name=f"batches/{index}", display_name=self.created[index]["display_name"],
                              state="JOB_STATE_SUCCEEDED", dest=types.BatchJobDestination(file_name=f"files/output-{index}"))

    def get(self, *, name):
        return self.job(int(name.split("/")[-1]))

    def download(self, *, file):
        job = self.created[int(file.split("-")[-1])]
        # Return reversed order: correlation must use keys, never positions.
        return ("\n".join(json.dumps({"key": row["key"], **self.rows(row["request"])})
                          for row in reversed(self.uploads[job["src"]])) + "\n").encode()


def requests(count=3):
    values = [{"contents": [{"role": "user", "parts": [{"text": f"q{i}"}]}]} for i in range(count)]
    return {batch.request_key("gemini-test", request): request for request in values}


def test_download_and_reimport_use_keys_and_no_new_paid_job(tmp_path):
    client = FakeClient()
    wanted = requests()
    saved = {}
    def consume(key, row, replay):
        saved[key] = batch.response_text(row["response"])
        return True
    kwargs = dict(model="gemini-test", requests=wanted, artifact=tmp_path / "answers.jsonl", consume=consume, client=client)
    batch.run_batch(**kwargs)
    assert saved == {key: request["contents"][0]["parts"][0]["text"] for key, request in wanted.items()}
    batch.run_batch(**kwargs)
    assert client.batches.create.call_count == client.files.download.call_count == 1


@pytest.mark.parametrize("error", [ConnectionError("offline"), KeyboardInterrupt()])
def test_interruption_while_polling_resumes_same_job(tmp_path, error):
    client = FakeClient()
    client.batches.get.side_effect = error
    kwargs = dict(model="gemini-test", requests=requests(), artifact=tmp_path / "answers.jsonl",
                  consume=lambda *args: True, client=client)
    with pytest.raises(type(error)):
        batch.run_batch(**kwargs)
    client.batches.get.side_effect = client.get
    batch.run_batch(**kwargs)
    assert client.batches.create.call_count == 1


def test_accepted_submission_with_lost_reply_is_discovered(tmp_path):
    client = FakeClient()
    def lost_reply(**kwargs):
        client.create(**kwargs)
        raise ConnectionError("lost create response")
    client.batches.create.side_effect = lost_reply
    kwargs = dict(model="gemini-test", requests=requests(), artifact=tmp_path / "answers.jsonl",
                  consume=lambda *args: True, client=client)
    with pytest.raises(ConnectionError):
        batch.run_batch(**kwargs)
    batch.run_batch(**kwargs)
    assert client.batches.create.call_count == 1
    assert client.batches.list.call_count == 1


def test_unconfirmed_submission_never_blindly_resubmits(tmp_path):
    client = FakeClient()
    client.batches.create.side_effect = ConnectionError("unknown submission")
    kwargs = dict(model="gemini-test", requests=requests(), artifact=tmp_path / "answers.jsonl",
                  consume=lambda *args: True, client=client)
    with pytest.raises(ConnectionError):
        batch.run_batch(**kwargs)
    with pytest.raises(RuntimeError, match="Submission outcome unknown"):
        batch.run_batch(**kwargs)
    assert client.batches.create.call_count == 1
    client.batches.create.side_effect = client.create
    batch.run_batch(**kwargs, retry_unconfirmed=True)
    assert client.batches.create.call_count == 2


def test_partial_failures_retry_only_failed_keys_on_next_run(tmp_path):
    client = FakeClient()
    wanted = requests()
    bad = next(iter(wanted))
    client.rows = lambda request: {"error": {"code": 503}} if batch.request_key("gemini-test", request) == bad else {"response": response()}
    saved = set()
    def consume(key, row, replay):
        if row.get("error"):
            return False
        saved.add(key)
        return True
    kwargs = dict(model="gemini-test", artifact=tmp_path / "answers.jsonl", consume=consume, client=client)
    batch.run_batch(requests=wanted, **kwargs)
    assert len(saved) == 2
    assert len(client.created) == 1
    client.rows = lambda request: {"response": response()}
    batch.run_batch(requests={bad: wanted[bad]}, **kwargs)
    assert len(saved) == 3
    assert [row["key"] for row in client.uploads[client.created[-1]["src"]]] == [bad]


def test_downloaded_results_survive_checkpoint_write_failure(tmp_path):
    client = FakeClient()
    kwargs = dict(model="gemini-test", requests=requests(), artifact=tmp_path / "answers.jsonl", client=client)
    with pytest.raises(OSError, match="disk full"):
        batch.run_batch(**kwargs, consume=Mock(side_effect=OSError("disk full")))
    batch.run_batch(**kwargs, consume=lambda *args: True)
    assert client.batches.create.call_count == client.files.download.call_count == 1


def test_small_failure_set_uses_sync_and_replays_cached_reply_after_disk_failure(tmp_path):
    client = FakeClient()
    wanted = requests(500)
    bad = list(wanted)[-1]
    client.rows = lambda request: ({"error": {"message": "failed"}} if batch.request_key("gemini-test", request) == bad
                                   else {"response": response()})
    saved = set()
    def consume(key, row, replay):
        if row.get("execution_mode") == "sync_fallback" and not replay:
            raise OSError("disk full")
        if row.get("error"):
            return False
        saved.add(key)
        return True
    kwargs = dict(model="gemini-test", artifact=tmp_path / "answers.jsonl", consume=consume,
                  client=client, sync_fallback_max=9)
    with pytest.raises(OSError, match="disk full"):
        batch.run_batch(requests=wanted, **kwargs)
    assert len(saved) == 499
    batch.run_batch(requests={bad: wanted[bad]}, **kwargs)
    assert saved == set(wanted)
    assert client.models.generate_content.call_count == 1
    assert client.batches.create.call_count == 1


@pytest.mark.parametrize("count,maximum,expected", [(9, 9, 9), (10, 9, 0), (1, 0, 0)])
def test_sync_fallback_threshold_counts_only_failed_items(tmp_path, count, maximum, expected):
    client = FakeClient()
    wanted = requests(count + 2)
    bad = set(list(wanted)[:count])
    client.rows = lambda request: ({"error": {"message": "failed"}} if batch.request_key("gemini-test", request) in bad
                                   else {"response": response()})
    batch.run_batch(model="gemini-test", requests=wanted, artifact=tmp_path / "answers.jsonl",
                    consume=lambda key, row, replay: not row.get("error"), client=client, sync_fallback_max=maximum)
    assert client.models.generate_content.call_count == expected
    assert client.batches.create.call_count == 1


def test_fallback_never_replaces_pending_or_unconfirmed_batch(tmp_path):
    client = FakeClient()
    client.batches.get.side_effect = lambda **kwargs: types.BatchJob(name="batches/0", state="JOB_STATE_RUNNING")
    kwargs = dict(model="gemini-test", requests=requests(), artifact=tmp_path / "answers.jsonl",
                  consume=lambda *args: True, client=client, sync_fallback_max=9, wait=False)
    with pytest.raises(batch.GeminiBatchPending):
        batch.run_batch(**kwargs)
    client.models.generate_content.assert_not_called()
    client.batches.create.side_effect = ConnectionError("unknown")
    kwargs["artifact"] = tmp_path / "other.jsonl"
    with pytest.raises(ConnectionError):
        batch.run_batch(**kwargs)
    with pytest.raises(RuntimeError, match="Submission outcome unknown"):
        batch.run_batch(**kwargs)
    client.models.generate_content.assert_not_called()


def test_fallback_budget_is_shared_by_old_and_new_batches(tmp_path):
    client = FakeClient()
    client.rows = lambda request: {"error": {"message": "failed"}}
    kwargs = dict(model="gemini-test", artifact=tmp_path / "answers.jsonl",
                  consume=lambda key, row, replay: not row.get("error"), client=client)
    batch.run_batch(requests=requests(2), **kwargs)
    batch.run_batch(requests=requests(4), sync_fallback_max=3, **kwargs)
    assert client.models.generate_content.call_count == 2
    assert client.batches.create.call_count == 2


@pytest.mark.parametrize("error,expected", [(ConnectionError("offline"), 3), (TypeError("invalid SDK arguments"), 1)])
def test_failed_sync_fallback_is_bounded_and_does_not_submit_another_batch(tmp_path, error, expected):
    client = FakeClient()
    client.rows = lambda request: {"error": {"message": "failed"}}
    client.models.generate_content.side_effect = error
    batch.run_batch(model="gemini-test", requests=requests(9), artifact=tmp_path / "answers.jsonl",
                    consume=lambda key, row, replay: not row.get("error"), client=client, sync_fallback_max=9)
    assert client.models.generate_content.call_count == expected
    assert client.batches.create.call_count == 1


@pytest.mark.parametrize("error,attempts", [(ConnectionError("offline"), 3), (TypeError("bad request"), 1)])
def test_stopped_fallback_for_old_keys_defers_new_keys(tmp_path, error, attempts):
    client = FakeClient()
    client.rows = lambda request: {"error": {"message": "failed"}}
    kwargs = dict(model="gemini-test", artifact=tmp_path / "answers.jsonl", client=client,
                  consume=lambda key, row, replay: "error" not in row)
    batch.run_batch(requests=requests(3), **kwargs)
    client.models.generate_content.side_effect = error
    batch.run_batch(requests=requests(4), sync_fallback_max=3, **kwargs)
    assert client.models.generate_content.call_count == attempts
    assert client.batches.create.call_count == client.files.upload.call_count == 1
    client.models.generate_content.side_effect = client.generate_content
    client.rows = lambda request: {"response": response()}
    batch.run_batch(requests=requests(4), sync_fallback_max=3, **kwargs)
    assert client.batches.create.call_count == 2
    assert [row["key"] for row in client.uploads["files/input-1"]] == list(requests(4))[-1:]


@pytest.mark.parametrize("failure", ["empty", "MAX_TOKENS"])
def test_recovered_generation_keeps_original_prompt_and_can_remain_incomplete(tmp_path, monkeypatch, failure):
    client = FakeClient()
    bad_response = response("")
    if failure != "empty":
        bad_response["candidates"][0]["finishReason"] = failure
    client.rows = lambda request: {"response": bad_response}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    engine = EvaluationEngine(language="english", initialize_judge=False)
    kwargs = dict(engine=engine, questions=["q1"], model="gemini-test", system="same system",
                  artifact=tmp_path / "answers.jsonl")
    assert generate_gemini_batch(**kwargs).aborted
    client.models.generate_content.side_effect = lambda **kwargs: types.GenerateContentResponse.model_validate(bad_response)
    assert generate_gemini_batch(**kwargs, sync_fallback_max=9).aborted
    assert client.batches.create.call_count == 1
    client.models.generate_content.side_effect = client.generate_content
    result = generate_gemini_batch(**kwargs, sync_fallback_max=9)
    assert not result.aborted
    assert result[0]["execution_mode"] == "sync_fallback"
    assert result[0]["batch_job"] == "batches/0"
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-test"
    assert call["contents"][0]["parts"][0]["text"] == "q1"
    assert call["config"].system_instruction.parts[0].text == "same system"
    assert client.batches.create.call_count == 1


def test_judge_fallback_preserves_schema_seed_and_scoring(tmp_path, monkeypatch):
    client = FakeClient()
    client.rows = lambda request: {"response": response("invalid JSON")}
    schema = EvaluationResultEnglish
    rubric = {section: {name: 5 for name, field in model.annotation.model_fields.items() if field.is_required()}
              for section, model in schema.model_fields.items()}
    client.models.generate_content.side_effect = lambda **kwargs: types.GenerateContentResponse.model_validate(response(json.dumps(rubric)))
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    engine = EvaluationEngine(model="gemini-test", language="english", initialize_judge=False)
    question, answer = "What is grace?", "God's favor in Christ. Ephesians 2:8-9."
    records = []
    result = judge_gemini_batch(engine, [(question, answer)], artifact=tmp_path / "results.jsonl",
                               on_result=records.append, sync_fallback_max=9)
    assert len(records) == 2 and "error" in records[0]
    assert result[0]["execution_mode"] == "sync_fallback"
    assert result[0]["evaluation"] == engine.process_evaluation(question, answer, schema.model_validate(rubric).model_dump())
    call = client.models.generate_content.call_args.kwargs
    original = next(iter(client.uploads.values()))[0]["request"]
    assert call["contents"] == original["contents"]
    assert call["config"].response_json_schema == original["generationConfig"]["responseJsonSchema"]
    assert call["config"].seed == engine.seed
    assert call["config"].system_instruction.parts[0].text == engine.system_prompt


@pytest.mark.parametrize("state", ["JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"])
def test_terminal_job_without_output_stays_incomplete_and_can_retry(tmp_path, state):
    client = FakeClient()
    client.batches.get.side_effect = lambda **kwargs: types.BatchJob(name="batches/0", state=state)
    records = []
    def consume(key, row, replay):
        records.append(row)
        return not row.get("error")
    kwargs = dict(model="gemini-test", requests=requests(), artifact=tmp_path / "answers.jsonl",
                  consume=consume, client=client)
    batch.run_batch(**kwargs)
    assert len(records) == 3
    assert all(state in row["error"]["message"] for row in records)
    assert client.batches.create.call_count == 1
    client.files.download.assert_not_called()
    client.batches.get.side_effect = client.get
    batch.run_batch(**kwargs)
    assert client.batches.create.call_count == 2
    assert all("response" in row for row in records[-3:])


def test_missing_output_retries_only_missing_key(tmp_path):
    client = FakeClient()
    wanted = requests()
    missing = next(iter(wanted))
    def incomplete_download(**kwargs):
        rows = [json.loads(line) for line in client.download(**kwargs).splitlines()]
        return b"\n".join(json.dumps(row).encode() for row in rows if row["key"] != missing)
    client.files.download.side_effect = incomplete_download
    saved = set()
    def consume(key, row, replay):
        if row.get("error"):
            return False
        saved.add(key)
        return True
    kwargs = dict(model="gemini-test", artifact=tmp_path / "answers.jsonl", consume=consume, client=client)
    batch.run_batch(requests=wanted, **kwargs)
    assert saved == set(wanted) - {missing}
    client.files.download.side_effect = client.download
    batch.run_batch(requests={missing: wanted[missing]}, **kwargs)
    assert saved == set(wanted)
    assert [row["key"] for row in client.uploads[client.created[-1]["src"]]] == [missing]


@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_output_keys_fail_before_import_or_resubmission(tmp_path, duplicate):
    client = FakeClient()
    wanted = requests()
    row = {"key": next(iter(wanted)) if duplicate else "unknown", "response": response()}
    client.files.download.side_effect = lambda **kwargs: (json.dumps(row) + "\n").encode() * (2 if duplicate else 1)
    consume = Mock(return_value=True)
    kwargs = dict(model="gemini-test", requests=wanted, artifact=tmp_path / "answers.jsonl",
                  consume=consume, client=client)
    for _ in range(2):
        with pytest.raises(ValueError, match="Unexpected or duplicate"):
            batch.run_batch(**kwargs)
    consume.assert_not_called()
    assert client.batches.create.call_count == client.files.download.call_count == 1


def test_nonblocking_batch_can_resume_and_blocks_switch_to_sync(tmp_path):
    client = FakeClient()
    client.batches.get.side_effect = lambda **kwargs: types.BatchJob(name="batches/0", state="JOB_STATE_PENDING")
    artifact = tmp_path / "answers.jsonl"
    kwargs = dict(model="gemini-test", requests=requests(), artifact=artifact,
                  consume=lambda *args: True, client=client, wait=False)
    with pytest.raises(batch.GeminiBatchPending):
        batch.run_batch(**kwargs)
    with pytest.raises(RuntimeError, match="resume with --gemini-execution batch"):
        batch.ensure_no_pending_batch(artifact)
    client.batches.get.side_effect = client.get
    batch.run_batch(**kwargs)
    batch.ensure_no_pending_batch(artifact)
    assert client.batches.create.call_count == 1


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_real_sdk_file_batch_uses_beta_and_does_not_retry_submission(tmp_path, monkeypatch, failed, fallback):
    import httpx2
    from google import genai
    from google.genai.errors import ServerError
    real_client = genai.Client
    http_requests = []
    uploaded_rows = []
    display_name = ""
    def handle(request):
        nonlocal display_name
        http_requests.append(request)
        if request.url.path.endswith("/files"):
            return httpx2.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/offline-upload"}, json={})
        if request.url.path == "/offline-upload":
            uploaded_rows.extend(json.loads(line) for line in request.content.decode().splitlines())
            return httpx2.Response(200, headers={"x-goog-upload-status": "final"}, json={"file": {"name": "files/input", "state": "ACTIVE"}})
        if request.url.path.endswith(":batchGenerateContent"):
            assert request.url.path == "/v1beta/models/gemini-test:batchGenerateContent"
            body = json.loads(request.content)
            assert body["batch"]["inputConfig"]["fileName"] == "files/input"
            display_name = body["batch"]["displayName"]
            if failed:
                return httpx2.Response(503, json={"error": {"code": 503, "message": "unavailable"}})
            return httpx2.Response(200, json={"name": "batches/job", "metadata": {"displayName": display_name, "state": "BATCH_STATE_PENDING"}})
        if request.url.path.endswith("/batches/job"):
            return httpx2.Response(200, json={"name": "batches/job", "metadata": {
                "displayName": display_name, "state": "BATCH_STATE_SUCCEEDED", "output": {"responsesFile": "files/output"},
            }})
        if request.url.path.endswith("files/output:download"):
            return httpx2.Response(200, content=("\n".join(json.dumps({"key": row["key"],
                **({"error": {"message": "missing response"}} if fallback and i == 0 else {"response": response()})})
                for i, row in enumerate(uploaded_rows)) + "\n").encode())
        if request.url.path.endswith(":generateContent"):
            assert request.url.path == "/v1beta/models/gemini-test:generateContent"
            body = json.loads(request.content)
            original = uploaded_rows[0]["request"]
            assert body["contents"] == original["contents"]
            assert body["systemInstruction"]["parts"] == original["systemInstruction"]["parts"]
            assert body["generationConfig"] == original["generationConfig"]
            return httpx2.Response(200, json=response())
        pytest.fail(f"Unexpected SDK request: {request.method} {request.url}")

    def client(**kwargs):
        options = kwargs.pop("http_options")
        options.client_args = {"transport": httpx2.MockTransport(handle)}
        options.retry_options.initial_delay = options.retry_options.max_delay = options.retry_options.jitter = 0
        return real_client(**kwargs, http_options=options)
    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    for name in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    raw_requests = list(requests().values())
    for request in raw_requests:
        request["systemInstruction"] = {"parts": [{"text": "original system prompt"}]}
        request["generationConfig"] = {"seed": 7, "responseMimeType": "application/json",
                                       "responseJsonSchema": {"type": "object", "properties": {"score": {"type": "integer"}}}}
    kwargs = dict(model="gemini-test", requests={batch.request_key("gemini-test", r): r for r in raw_requests},
                  artifact=tmp_path / "answers.jsonl", consume=lambda key, row, replay: not row.get("error"),
                  sync_fallback_max=9 if fallback else 0)
    if failed:
        with pytest.raises(ServerError):
            batch.run_batch(**kwargs)
    else:
        batch.run_batch(**kwargs)
    assert sum(request.url.path.endswith(":batchGenerateContent") for request in http_requests) == 1
    assert sum(request.url.path.endswith(":generateContent") for request in http_requests) == int(fallback and not failed)


@pytest.mark.parametrize("change", ["answer", "system", "model"])
def test_request_identity_changes_with_evaluation_inputs(change):
    original = {"contents": [{"parts": [{"text": "question and answer"}]}], "systemInstruction": {"parts": [{"text": "system"}]}}
    changed = json.loads(json.dumps(original))
    model = "gemini-test"
    if change == "answer": changed["contents"][0]["parts"][0]["text"] += " changed"
    if change == "system": changed["systemInstruction"]["parts"][0]["text"] += " changed"
    if change == "model": model = "gemini-other"
    assert batch.request_key(model, changed) != batch.request_key("gemini-test", original)


def test_generation_adapter_preserves_checkpoint_metadata_and_rejects_truncation(tmp_path, monkeypatch):
    client = FakeClient()
    client.rows = lambda request: {"response": response("answer " + request["contents"][0]["parts"][0]["text"])}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    engine = EvaluationEngine(language="english", initialize_judge=False)
    records = []
    result = generate_gemini_batch(engine, ["q1", "q2"], model="gemini-test", system="system",
                                    artifact=tmp_path / "answers.jsonl", on_result=records.append)
    assert not result.aborted
    assert [(r["question"], r["answer"]) for r in records] == [("q1", "answer q1"), ("q2", "answer q2")]
    assert records[0]["usage_metadata"]["thoughtsTokenCount"] == 7
    assert all(row["request"]["systemInstruction"]["parts"][0]["text"] == "system"
               for row in next(iter(client.uploads.values())))
    truncated = response()
    truncated["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        batch.response_text(truncated)


def test_judging_adapter_uses_existing_rubric_and_heuristics(tmp_path, monkeypatch):
    client = FakeClient()
    schema = EvaluationResultEnglish
    rubric = {section: {name: 5 for name, field in model.annotation.model_fields.items() if field.is_required()}
              for section, model in schema.model_fields.items()}
    client.rows = lambda request: {"response": response(json.dumps(rubric))}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    engine = EvaluationEngine(model="gemini-test", language="english", initialize_judge=False)
    question, answer = "What is grace?", "God's favor in Christ. Ephesians 2:8-9."
    result = judge_gemini_batch(engine, [(question, answer)], artifact=tmp_path / "results.jsonl")
    assert result[0]["evaluation"] == engine.process_evaluation(question, answer, schema.model_validate(rubric).model_dump())
    request = next(iter(client.uploads.values()))[0]["request"]
    assert request["generationConfig"]["responseJsonSchema"] == schema.model_json_schema()
    assert request["contents"][0]["parts"][0]["text"] == engine.instructions + "\n\n" + engine.evaluation_content(question, answer)


def test_cli_generation_then_judging_batch_reuses_all_saved_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient()
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    questions = tmp_path / "questions.txt"
    questions.write_text("q1\nq2\n")
    dataset = tmp_path / "answers.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    args = ["--language", "english", "--limit", "0", "--questions-file", str(questions),
            "--gemini-execution", "batch", "--no-progress", "--require-complete-answers"]
    generate = ["--mode", "generate-api_evals", "--generate-only", "--provider", "google",
                "--gen-model", "gemini-test", "--output-dataset", str(dataset), *args]
    client.batches.get.side_effect = lambda **kwargs: types.BatchJob(name="batches/0", state="JOB_STATE_PENDING")
    assert cli.main([*generate, "--gemini-batch-no-wait"]) == 75
    assert not dataset.exists()
    client.batches.get.side_effect = client.get
    assert cli.main(generate) == 0
    original = dataset.read_bytes()
    assert cli.main(generate) == 0
    assert dataset.read_bytes() == original
    rubric = {section: {name: 4 for name, field in model.annotation.model_fields.items() if field.is_required()}
              for section, model in EvaluationResultEnglish.model_fields.items()}
    client.rows = lambda request: {"response": response(json.dumps(rubric))}
    judge = ["--mode", "dataset", "--dataset", str(dataset), "--judge-model", "gemini-test",
             "--answers-label", "answers", "--results-jsonl", str(reviews), "--skip-comparison-csv", *args]
    assert cli.main(judge) == 0
    original_reviews = reviews.read_bytes()
    assert cli.main(judge) == 0
    assert reviews.read_bytes() == original_reviews
    assert len(client.created) == 2


@pytest.mark.parametrize("stage", ["generating", "judging"])
@pytest.mark.parametrize("resume_execution", ["sync", "batch"])
def test_cli_reconciles_interruption_after_final_batch_checkpoint(tmp_path, monkeypatch, stage, resume_execution):
    monkeypatch.chdir(tmp_path)
    questions = tmp_path / "questions.txt"
    questions.write_text("q\n")
    dataset = tmp_path / "answers.jsonl"
    artifact = dataset if stage == "generating" else tmp_path / "reviews.jsonl"
    args = ["--language", "english", "--limit", "0", "--questions-file", str(questions),
            "--gemini-execution", "batch", "--no-progress", "--skip-comparison-csv"]
    client = FakeClient()
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    if stage == "generating":
        args.extend(["--mode", "generate-api_evals", "--generate-only", "--provider", "google",
                     "--gen-model", "gemini-test", "--output-dataset", str(dataset)])
    else:
        dataset.write_text(json.dumps({"messages": [{"role": "user", "content": "q"},
                                                   {"role": "assistant", "content": "answer"}]}) + "\n")
        rubric = {section: {name: 4 for name, field in model.annotation.model_fields.items() if field.is_required()}
                  for section, model in EvaluationResultEnglish.model_fields.items()}
        client.rows = lambda request: {"response": response(json.dumps(rubric))}
        args.extend(["--mode", "dataset", "--dataset", str(dataset), "--answers-label", "answers",
                     "--judge-model", "gemini-test", "--results-jsonl", str(artifact)])

    save = batch._save
    def interrupt_final_save(path, value):
        if value.get("phase") == "imported":
            raise KeyboardInterrupt
        save(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(batch, "_save", interrupt_final_save)
        with pytest.raises(KeyboardInterrupt):
            cli.main(args)
    original = artifact.read_bytes()
    manifest = next(batch.state_directory(artifact).glob("*.job.json"))
    pending = json.loads(manifest.read_text())
    assert pending["phase"] == "submitted"
    artifact.write_text(json.dumps({"request_key": pending["keys"][0], "error": "unfinished"}) + "\n")
    with pytest.raises(RuntimeError, match="unfinished Gemini batch"):
        batch.ensure_no_pending_batch(artifact)
    assert json.loads(manifest.read_text())["phase"] == "submitted"
    artifact.write_bytes(original)
    monkeypatch.setattr(batch, "_new_client", Mock(side_effect=AssertionError("unexpected provider call")))
    engine = EvaluationEngine(model="gemini-test", language="english", initialize_judge=False)
    monkeypatch.setattr(cli, "EvaluationEngine", lambda **kwargs: engine)
    args[args.index("--gemini-execution") + 1] = resume_execution
    assert cli.main(args) == 0
    assert json.loads(manifest.read_text())["phase"] == "imported"
    batch.ensure_no_pending_batch(artifact)
    assert artifact.read_bytes() == original


def test_cli_resumes_499_saved_answers_using_one_sync_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeClient()
    blocked = response("")
    client.rows = lambda request: {"response": blocked if request["contents"][0]["parts"][0]["text"] == "q253" else response()}
    monkeypatch.setattr(batch, "_new_client", lambda: client)
    questions = tmp_path / "questions.txt"
    questions.write_text("\n".join(f"q{i}" for i in range(1, 501)))
    dataset = tmp_path / "answers.jsonl"
    args = ["--language", "english", "--limit", "0", "--questions-file", str(questions),
            "--gemini-execution", "batch", "--no-progress", "--require-complete-answers",
            "--mode", "generate-api_evals", "--generate-only", "--provider", "google",
            "--gen-model", "gemini-test", "--output-dataset", str(dataset), "--use-system-prompt"]
    assert cli.main(args) == 1
    original = dataset.read_bytes()
    assert len(cli.load_qa_pairs(str(dataset), limit=0)) == 499
    assert cli.main([*args, "--gemini-batch-sync-fallback-max", "9"]) == 0
    assert dataset.read_bytes().startswith(original)
    assert len(cli.load_qa_pairs(str(dataset), limit=0)) == 500
    assert client.batches.create.call_count == client.models.generate_content.call_count == 1
    assert client.models.generate_content.call_args.kwargs["contents"][0]["parts"][0]["text"] == "q253"
    completed = dataset.read_bytes()
    assert cli.main([*args, "--gemini-batch-sync-fallback-max", "9"]) == 0
    assert dataset.read_bytes() == completed
    assert client.batches.create.call_count == client.models.generate_content.call_count == 1


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX process groups")
@pytest.mark.parametrize("stage", ["submission", "import499"])
@pytest.mark.parametrize("termination", [signal.SIGINT, signal.SIGKILL])
def test_process_restart_recovers_remote_batch_and_499_imported_answers(tmp_path, stage, termination):
    root = Path(__file__).resolve().parent.parent
    script = tmp_path / "run.py"
    script.write_text(f"import sys\nsys.path[:0] = {[str(root), str(root/'tests')]!r}\n" + '''
import json, os, time, socket
from pathlib import Path
from test_gemini_batch import FakeClient, requests
from parrot_ai.gemini_batch import run_batch
from cp_eval_llms import _append_jsonl_record
def denied(*args, **kwargs): raise AssertionError("No network in tests")
socket.socket.connect = denied
socket.socket.connect_ex = denied
client = FakeClient(Path("cloud.json"))
def pause():
    Path("ready").touch()
    while True: time.sleep(.05)
if os.environ.get("PAUSE") == "submission":
    def create(**kwargs):
        result = client.create(**kwargs)
        pause()
        return result
    client.batches.create.side_effect = create
artifact = Path("answers.jsonl")
saved = {json.loads(line)["key"] for line in artifact.read_text().splitlines()} if artifact.exists() else set()
wanted = {key: request for key, request in requests(500).items() if key not in saved}
def consume(key, row, replay):
    if os.environ.get("PAUSE") == "import499" and len(saved) == 499: pause()
    _append_jsonl_record(artifact, {"key": key, **row})
    saved.add(key)
    return True
run_batch(model="gemini-test", requests=wanted, artifact=artifact, consume=consume, client=client)
''')
    env = {**os.environ, "PAUSE": stage}
    process = subprocess.Popen([sys.executable, str(script)], cwd=tmp_path, env=env,
                               start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, "Worker exited before reaching checkpoint"
            assert time.monotonic() < deadline
            time.sleep(.02)
        os.killpg(process.pid, termination)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    artifact = tmp_path / "answers.jsonl"
    before = artifact.read_bytes() if artifact.exists() else b""
    assert len(before.splitlines()) == (499 if stage == "import499" else 0)
    env.pop("PAUSE")
    completed = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert artifact.read_bytes().startswith(before)
    assert len(artifact.read_text().splitlines()) == 500
    assert len(json.loads((tmp_path / "cloud.json").read_text())["created"]) == 1
