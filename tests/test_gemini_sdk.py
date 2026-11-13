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
            return httpx2.Response(200, json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": response["text"]}]},
                "finishReason": "STOP",
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
        options.client_args = {"transport": httpx2.MockTransport(handle)}
        client = real_client(**kwargs, http_options=options)
        clients.append(client)
        return client

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", "httpx2")
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
