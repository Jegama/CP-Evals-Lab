"""Offline compatibility checks using Together's actual SDK and HTTP transport."""

import json

import httpx
import pytest
import together

from parrot_ai.core import ParrotAITogether
from parrot_ai.llm_evaluation import EvaluationEngine


@pytest.fixture
def together_transport(monkeypatch):
    requests = []
    response = {"status": 200}
    clients = []
    real_client = together.Together

    def handle(request):
        requests.append(request)
        if response["status"] != 200:
            return httpx.Response(response["status"], json={"error": {"message": "Offline failure"}})
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Offline answer",
            }}],
        })

    def make_client(**kwargs):
        client = real_client(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)), max_retries=0,
        )
        clients.append(client)
        return client

    monkeypatch.setattr("parrot_ai.core.load_dotenv", lambda: None)
    monkeypatch.setenv("TOGETHER_API_KEY", "offline-test-key")
    for name in ("TOGETHER_BASE_URL", "TOGETHER_PROJECT_ID", "TOGETHER_CUSTOM_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(together, "Together", make_client)
    yield requests, response
    for client in clients:
        client.close()


@pytest.mark.parametrize("system", [None, "Explicit system prompt", ""])
def test_together_v3_generation_uses_inference_route_and_preserves_prompt_opt_in(together_transport, system):
    requests, _ = together_transport
    wrapper = ParrotAITogether(language="english")

    assert wrapper.generate("Question", system=system, model="test/model") == "Offline answer"

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api-inference.together.ai/v1/chat/completions"
    body = json.loads(requests[0].content)
    expected = [{"role": "user", "content": "Question"}]
    if system is not None:
        expected.insert(0, {"role": "system", "content": system or wrapper.prompts.MAIN_SYSTEM_PROMPT})
    assert body == {"model": "test/model", "messages": expected}


@pytest.mark.parametrize("status,attempts", [(401, 1), (503, 3)])
def test_together_v3_errors_preserve_batch_failure_cutoffs(together_transport, status, attempts):
    requests, response = together_transport
    response["status"] = status
    engine = object.__new__(EvaluationEngine)
    engine.language = "english"

    results = engine.generate_responses(
        ["Question"] * 10, provider="together", model="test/model", progress=False,
    )

    assert results.aborted
    assert len(requests) == len(results) == attempts
    assert all("error" in result and "answer" not in result for result in results)
