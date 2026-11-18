"""Bridge remote Gemini batches to the normal per-question checkpoints."""

from pathlib import Path

from ..evaluation_schemas import EvaluationResultArabic, EvaluationResultEnglish
from ..gemini_batch import ensure_no_pending_batch, parse_batch_response, request_key, run_batch
from ..generation_failures import TerminalGenerationFailure
from ..llm_evaluation import GenerationBatch
from .progress_reporting import emit_progress


def _request(contents: str, system: str | None) -> dict:
    request = {"contents": [{"role": "user", "parts": [{"text": contents}]}]}
    if system:
        request["systemInstruction"] = {"parts": [{"text": system}]}
    return request


def _review_schema(engine, stage):
    if stage == "judging":
        return EvaluationResultArabic if engine.language == "arabic" else EvaluationResultEnglish
    return None


def _item_requests(engine, items, model, system, schema):
    json_schema = schema.model_json_schema() if schema else None
    for index, (question, answer) in enumerate(items):
        if schema:
            if not answer.strip():
                raise ValueError("Cannot batch-judge an empty answer")
            content = f"{engine.instructions}\n\n{engine.evaluation_content(question, answer)}"
            request = _request(content, engine.system_prompt)
            request["generationConfig"] = {
                "seed": engine.seed, "responseMimeType": "application/json",
                "responseJsonSchema": json_schema,
            }
        else:
            request = _request(question, system)
        yield request_key(model, request), request, (index, question, answer)


def guard_gemini_sync(engine, items, *, model, artifact, stage, system=None):
    schema = _review_schema(engine, stage)
    pending_keys = (key for key, _, _ in _item_requests(engine, items, model, system, schema))
    ensure_no_pending_batch(Path(artifact), pending_keys=pending_keys, review_schema=schema)


def _run(
    engine, items, *, model, artifact, stage, on_result=None,
    system=None, progress_initial=0, progress_total=None, poll_seconds=30,
    retry_unconfirmed=False, wait=True, sync_fallback_max=0,
):
    schema = _review_schema(engine, stage)
    requests = {}
    identities = {}
    for key, request, identity in _item_requests(engine, items, model, system, schema):
        requests[key] = request
        identities[key] = identity
    results = {}
    total = progress_total if progress_total is not None else progress_initial + len(items)

    def consume(key, row, replay):
        index, question, answer = identities[key]
        response = row.get("response") or {}
        record = {
            "index": index, "question": question,
            "execution_mode": row.get("execution_mode", "batch"),
            "batch_job": row.get("batch_job"), "request_key": key,
            "usage_metadata": response.get("usageMetadata", response.get("usage_metadata", {})),
        }
        execution_label = "sync fallback" if record["execution_mode"] == "sync_fallback" else "batch"
        try:
            if row.get("error"):
                raise ValueError(f"Gemini {execution_label} request failed: {row['error']}")
            parsed = parse_batch_response(response, schema)
            if stage == "judging":
                record.update(answer=answer, evaluation=engine.process_evaluation(question, answer, parsed))
            else:
                record.update(answer=parsed, provider="google")
        except (ValueError, TypeError, KeyError) as exc:
            if stage == "generating" and isinstance(exc, TerminalGenerationFailure):
                record.update(answer="", provider="google", generation_failure=exc.metadata())
                print(f"[model failure] Generation question {index + 1}: {exc.reason}; recorded for minimum scoring, no retry")
            else:
                if replay:
                    return False  # A fresh run may retry this failed request.
                record["error"] = str(exc)
                print(f"[error] {stage} {execution_label} question {index + 1}: {exc}")
        # Disk failures must propagate, keeping the job available for import.
        if on_result:
            on_result(record)
        results[key] = record
        saved = sum("error" not in item for item in results.values())
        emit_progress(stage, progress_initial + saved, total, reused=progress_initial)
        return "error" not in record

    def waiting(state):
        saved = sum("error" not in item for item in results.values())
        emit_progress(f"{stage} batch {state.removeprefix('JOB_STATE_').lower()}",
                      progress_initial + saved, total, reused=progress_initial)

    run_batch(model=model, requests=requests, artifact=Path(artifact), consume=consume,
              on_wait=waiting, poll_seconds=poll_seconds, retry_unconfirmed=retry_unconfirmed,
              wait=wait, sync_fallback_max=sync_fallback_max)
    return list(results.values())


def generate_gemini_batch(engine, questions, *, model, artifact, **kwargs):
    results = GenerationBatch()
    results.extend(_run(engine, [(q, "") for q in questions], model=model,
                        artifact=artifact, stage="generating", **kwargs))
    results.aborted = len(results) != len(questions) or any("error" in row for row in results)
    return results


def judge_gemini_batch(engine, pairs, *, artifact, **kwargs):
    return _run(engine, pairs, model=engine.model, artifact=artifact, stage="judging", **kwargs)
