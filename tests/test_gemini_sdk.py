"""Exercise Gemini request serialization with the installed SDK, entirely offline."""

import json

import httpx2
import pytest
from google import genai
from google.genai import types
from pydantic import BaseModel

from parrot_ai.core import ParrotAIGemini
from parrot_ai.evaluation_schemas import EvaluationResultArabic, EvaluationResultEnglish
from parrot_ai.llm_evaluation import EvaluationEngine
from parrot_ai.gemini_transport import gemini_http_options


class StructuredAnswer(BaseModel):
    answer: str


@pytest.fixture
def gemini_transport(monkeypatch):
    requests = []
    response = {"text": "Offline answer"}
    clients = []
    real_client = genai.Client
    file_data = {
        "name": "files/offline-audio",
        "uri": "https://generativelanguage.googleapis.com/v1/files/offline-audio",
        "mimeType": "audio/mpeg",
        "state": "ACTIVE",
    }

    def handle(request):
        requests.append(request)
        if request.url.path.endswith(":generateContent"):
            if response.get("failures"):
                failure = response["failures"].pop(0)
                if isinstance(failure, Exception):
                    raise failure
                return httpx2.Response(failure, json={"error": {
                    "code": failure, "message": "Offline temporary failure",
                }})
            return httpx2.Response(200, json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": response["text"]}]},
                "finishReason": response.get("finish_reason", "STOP"),
            }]})
        if request.url.path.endswith("/files"):
            return httpx2.Response(200, headers={
                "x-goog-upload-url": "https://generativelanguage.googleapis.com/offline-upload",
            }, json={})
        if request.url.path == "/offline-upload":
            return httpx2.Response(200, headers={"x-goog-upload-status": "final"}, json={"file": file_data})
        if request.url.path.endswith("/files/offline-audio"):
            return httpx2.Response(200, json=file_data)
        pytest.fail(f"Unexpected SDK request: {request.method} {request.url.path}")

    def make_client(**kwargs):
        options = types.HttpOptions.model_validate(kwargs.pop("http_options", {}))
        if options.retry_options:
            options.retry_options.initial_delay = 0
            options.retry_options.max_delay = 0
            options.retry_options.jitter = 0
        options.client_args = {"transport": httpx2.MockTransport(handle)}
        client = real_client(**kwargs, http_options=options)
        clients.append(client)
        return client

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
    monkeypatch.delenv("CP_GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CP_GEMINI_MAX_ATTEMPTS", raising=False)
    for name in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(genai, "Client", make_client)
    yield requests, response
    for client in clients:
        client.close()


@pytest.mark.parametrize("method", [
    "generate", "generate_with_contents", "generate_structured", "generate_structured_with_contents",
])
def test_generation_uses_stable_endpoint_and_serializes_schema(gemini_transport, method):
    requests, response = gemini_transport
    wrapper = ParrotAIGemini(language="english")
    structured = "structured" in method
    if structured:
        response["text"] = '{"answer": "Offline answer"}'
    contents = ["Question", types.Part.from_uri(
        file_uri="https://generativelanguage.googleapis.com/v1/files/offline-audio",
        mime_type="audio/mpeg",
    )] if "with_contents" in method else "Question"
    kwargs = {"response_schema": StructuredAnswer} if structured else {}

    result = getattr(wrapper, method)(contents, **kwargs)

    assert result == ({"answer": "Offline answer"} if structured else "Offline answer")
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models/gemini-3.8-flash:generateContent"
    body = json.loads(requests[0].content)
    assert "systemInstruction" not in body
    if structured:
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseSchema"]["required"] == ["answer"]
    if "with_contents" in method:
        file_data = types.FileData.model_validate(body["contents"][0]["parts"][1]["fileData"])
        assert file_data.mime_type == "audio/mpeg"


@pytest.mark.parametrize("language,schema", [
    ("english", EvaluationResultEnglish), ("arabic", EvaluationResultArabic),
])
def test_judging_uses_stable_endpoint_and_parses_rubric(gemini_transport, language, schema):
    requests, response = gemini_transport
    rubric = {
        section: {name: 4 for name, field in model.annotation.model_fields.items() if field.is_required()}
        for section, model in schema.model_fields.items()
    }
    response["text"] = schema.model_validate(rubric).model_dump_json()
    engine = EvaluationEngine(model="gemini-3.8-flash", language=language)

    result = engine.evaluate("What is grace?", "God's favor in Christ. Ephesians 2:8-9.")

    assert result["Adherence"]["Core"] == 4
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models/gemini-3.8-flash:generateContent"
    body = json.loads(requests[0].content)
    assert body["generationConfig"]["seed"] == 7
    assert "Adherence" in body["generationConfig"]["responseSchema"]["properties"]
    assert body["systemInstruction"]["parts"][0]["text"] == engine.system_prompt


def test_files_upload_and_lookup_use_stable_endpoint(gemini_transport, tmp_path):
    requests, _ = gemini_transport
    wrapper = ParrotAIGemini(language="english")
    audio = tmp_path / "sermon.mp3"
    audio.write_bytes(b"offline audio fixture")

    uploaded = wrapper.upload_file(str(audio))
    retrieved = wrapper.get_file("offline-audio")

    assert uploaded.name == retrieved.name == "files/offline-audio"
    assert [request.url.path for request in requests] == [
        "/upload/v1/files", "/offline-upload", "/v1/files/offline-audio",
    ]
    assert requests[1].content == audio.read_bytes()


def test_gemini_recovery_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("CP_GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CP_GEMINI_MAX_ATTEMPTS", raising=False)
    options = gemini_http_options()
    assert options.timeout == 120_000
    assert options.retry_options.attempts == 5
    assert options.retry_options.initial_delay == 5
    assert options.retry_options.max_delay == 30
    monkeypatch.setenv("CP_GEMINI_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("CP_GEMINI_MAX_ATTEMPTS", "3")
    assert gemini_http_options().timeout == 240_000
    assert gemini_http_options().retry_options.attempts == 3
    monkeypatch.setenv("CP_GEMINI_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="positive"):
        gemini_http_options()


@pytest.mark.parametrize("entrypoint", ["generate", "judge"])
@pytest.mark.parametrize("failure", [httpx2.ReadTimeout("stalled socket"), httpx2.ConnectError("offline"), 503, 429])
def test_gemini_recovers_same_request_after_network_interruption(gemini_transport, entrypoint, failure):
    requests, response = gemini_transport
    response["failures"] = [failure, failure]
    if entrypoint == "generate":
        assert ParrotAIGemini(language="english").generate("Question") == "Offline answer"
    else:
        schema = EvaluationResultEnglish
        response["text"] = schema.model_validate({
            section: {name: 4 for name, field in model.annotation.model_fields.items() if field.is_required()}
            for section, model in schema.model_fields.items()
        }).model_dump_json()
        engine = EvaluationEngine(model="gemini-4-hydrogen", language="english")
        assert engine.evaluate("What is grace?", "God's favor in Christ.")["Adherence"]["Core"] == 4
    assert len(requests) == 3
    assert all(request.content == requests[0].content for request in requests)
    assert all(request.extensions["timeout"]["read"] == 120 for request in requests)


@pytest.mark.parametrize("failure,attempts", [(503, 5), (401, 1), (400, 1)])
def test_gemini_recovery_is_bounded_and_does_not_retry_permanent_errors(gemini_transport, failure, attempts):
    requests, response = gemini_transport
    response["failures"] = [failure] * 10
    with pytest.raises(Exception):
        ParrotAIGemini(language="english").generate("Question")
    assert len(requests) == attempts


def test_gemini_socket_timeout_exhaustion_is_bounded(gemini_transport):
    requests, response = gemini_transport
    response["failures"] = [httpx2.ReadTimeout("stalled socket")] * 10
    with pytest.raises(httpx2.ReadTimeout):
        ParrotAIGemini(language="english").generate("Question")
    assert len(requests) == 5
