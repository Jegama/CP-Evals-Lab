"""Evaluation engine and public API for theological QA model assessment.

This module is the main entry point. Helper functions live in the
``parrot_ai.llm_evals`` sub-package and are re-exported here for
backward compatibility.

    from parrot_ai.llm_evaluation import EvaluationEngine
    engine = EvaluationEngine(model="gpt-5-mini")
    pairs = engine.load_qa_pairs("data/arabic/ar_training_dataset_small_model.jsonl")
    results = engine.batch_evaluate(pairs, limit=10)
"""

import os
import json
import importlib
from typing import List, Tuple, Dict, Any, Optional, Iterable, Callable, cast
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic
from tqdm import tqdm
from dotenv import load_dotenv

from .evaluation_schemas import (
    EvaluationResultArabic,
    EvaluationResultEnglish,
)
from .core import (
    ParrotAIOpenAI,
    ParrotAITogether,
    ParrotAIGemini,
    ParrotAIGrok,
    ParrotAIHF,
    ParrotAIClaude,
)

# Import all helpers from sub-package
from .llm_evals import (
    load_qa_pairs,
    load_eval_questions,
    apply_purity_penalty,
    clamp_scale_scores,
    clamp_all_overalls,
    enforce_knockouts,
    calibrate_english_scores,
    calibrate_arabic_scores,
    has_scripture_citation,
    has_theological_terminology,
    has_arabic_scripture_citation,
    has_arabic_theological_terminology,
)
from .llm_evals.progress_reporting import emit_progress
from .llm_evals.score_validation import has_rubric_scores, minimum_rubric_scores
from .gemini_transport import gemini_http_options
from .generation_failures import TerminalGenerationFailure

load_dotenv()

DEFAULT_MODEL = "gpt-5-mini"


def _stop_after_error(error: Exception, consecutive_errors: int) -> bool:
    """Stop broken SDK/configuration calls immediately, other failures after three."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        # Google Gen AI exposes the HTTP status as `code`.
        status_code = getattr(error, "code", None)
    return (
        isinstance(error, (TypeError, AttributeError))
        or status_code in (400, 401, 403, 404, 422)
        or consecutive_errors >= 3
    )


class GenerationBatch(list[dict]):
    """List-compatible attempted responses with an explicit failure-cutoff flag."""

    def __init__(self) -> None:
        super().__init__()
        self.aborted = False


class EvaluationEngine:
    """Encapsulates evaluation logic for reuse in notebooks or scripts.

    Temperature / max tokens omitted per project design for stability.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        language: str = "arabic",
        seed: Optional[int] = 7,
        initialize_judge: bool = True,
    ) -> None:
        """Create a language-aware evaluation engine.

        language: selects prompt module parrot_ai.prompts.<language>
        The module must define EVAL_SYSTEM_PROMPT and EVAL_INSTRUCTIONS.
        """
        self.language = language
        self.seed = seed
        self.model = model
        if not initialize_judge:
            # Generation must not depend on any judge's credentials or SDK setup.
            self.client = None
        elif model == DEFAULT_MODEL:
            self.client = client or OpenAI()
        elif model.startswith("gemini"):
            # Use the stable API consistently for judging and generation.
            self.client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
                http_options=gemini_http_options(),
            )
        elif model.startswith("claude"):
            self.client = anthropic.Anthropic()
        else:
            self.client = client or OpenAI()
            print(
                f"Warning: Using non-default model '{model}' with OpenAI client may cause issues if the model is not supported."
            )
        prompt_module_name = f"parrot_ai.prompts.{language}"
        self.prompts = importlib.import_module(prompt_module_name)
        required = ["EVAL_SYSTEM_PROMPT", "EVAL_INSTRUCTIONS"]
        missing = [r for r in required if not hasattr(self.prompts, r)]
        if missing:
            raise ValueError(
                f"Missing required evaluation prompt constants in {prompt_module_name}: {missing}"
            )
        self.system_prompt = getattr(self.prompts, "EVAL_SYSTEM_PROMPT")
        self.instructions = getattr(self.prompts, "EVAL_INSTRUCTIONS")

    # -------------- Core single evaluation --------------
    def evaluate(self, question: str, answer: str) -> dict:
        """Evaluate a single (question, answer) pair returning rubric dict."""
        # Fast path for empty / whitespace-only answers: assign all 1s (lowest rubric) + penalty
        if not answer.strip():
            return minimum_rubric_scores(self.language)
        user_content = self.evaluation_content(question, answer)
        # Choose response model based on language
        response_model = (
            EvaluationResultArabic
            if self.language == "arabic"
            else EvaluationResultEnglish
        )
        if self.model.startswith("gemini"):
            # Gemini path - cast types to Any to work around type stub limitations
            types_any = cast(Any, types)
            cfg = types_any.GenerateContentConfig(
                seed=self.seed,
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=response_model,
            )
            # Concatenate instructions and the Q/A payload into one content string
            contents = f"{self.instructions}\n\n{user_content}"
            # Use dynamic attribute access to avoid strict type issues across SDK versions
            client_any = cast(Any, self.client)
            resp = client_any.models.generate_content(
                model=self.model,
                contents=contents,
                config=cfg,
            )
            candidates = getattr(resp, "candidates", None) or []
            reason = candidates[0].finish_reason if candidates else None
            if reason != "STOP":
                raise ValueError(f"Gemini judge output did not finish successfully: {reason}")
            # Prefer structured parsed output; fallback to parsing JSON text
            result_dict: dict
            parsed_obj = getattr(resp, "parsed", None)
            if parsed_obj is not None:
                # If SDK returned a Pydantic model instance
                if hasattr(parsed_obj, "model_dump"):
                    result_dict = parsed_obj.model_dump()
                elif isinstance(parsed_obj, dict):
                    result_dict = parsed_obj
                else:
                    # Try use resp.text which should be a JSON string
                    try:
                        result_dict = json.loads(getattr(resp, "text", "") or "{}")
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Failed to parse Gemini evaluation result: {e}"
                        )
            else:
                try:
                    result_dict = json.loads(getattr(resp, "text", "") or "{}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse Gemini evaluation result: {e}")
        elif self.model.startswith("claude"):
            # Claude/Anthropic path using structured outputs
            client_any = cast(Any, self.client)
            messages = [
                {"role": "user", "content": f"{self.instructions}\n\n{user_content}"},
            ]
            parse_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": messages,
                "output_format": response_model,
            }
            if self.system_prompt:
                parse_kwargs["system"] = self.system_prompt
            response = client_any.messages.parse(**parse_kwargs)
            parsed = response.parsed_output
            if parsed is None:
                raise ValueError("Failed to parse structured Anthropic evaluation response")
            if hasattr(parsed, "model_dump"):
                result_dict = parsed.model_dump()
            elif isinstance(parsed, dict):
                result_dict = parsed
            else:
                result_dict = json.loads(parsed.model_dump_json())
        else:
            client_any = cast(Any, self.client)
            completion = client_any.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.instructions},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_model,
                seed=self.seed,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError(
                    "Failed to parse evaluation result from OpenAI response"
                )
            result_dict = json.loads(parsed.model_dump_json())
        return self.process_evaluation(question, answer, result_dict)

    def evaluation_content(self, question: str, answer: str) -> str:
        if self.language == "arabic":
            return f"السؤال:\n{question}\n\nالإجابة:\n{answer}\n\nقيّم وفق التعليمات السابقة."
        return f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvaluate per the rubric instructions above."

    def process_evaluation(self, question: str, answer: str, result_dict: dict) -> dict:
        """Apply the same score validation and ordered heuristics to either transport."""
        # Check before heuristics can create empty sections or coerce non-scores.
        if not has_rubric_scores(result_dict):
            raise ValueError("Judge returned no evaluation scores")
        result_dict = clamp_scale_scores(result_dict)
        # Only apply Arabic purity heuristics for Arabic language
        if self.language == "arabic" and "Arabic_Accuracy" in result_dict:
            result_dict = apply_purity_penalty(answer, result_dict)
        clamp_all_overalls(result_dict)
        # Apply language-specific calibration heuristics
        if self.language == "english":
            result_dict = calibrate_english_scores(question, answer, result_dict)
        elif self.language == "arabic":
            result_dict = calibrate_arabic_scores(question, answer, result_dict)
        result_dict = enforce_knockouts(answer, result_dict)
        clamp_all_overalls(result_dict)
        return result_dict

    # -------------- Batch utilities --------------
    def batch_evaluate(
        self,
        pairs: Iterable[Tuple[str, str]],
        limit: Optional[int] = None,
        progress: bool = True,
        stop_on_error: bool = False,
        on_result: Optional[Callable[[dict], None]] = None,
        progress_initial: int = 0,
        progress_total: Optional[int] = None,
    ) -> list[dict]:
        """Evaluate multiple QA pairs.

        Args:
            pairs: An iterable of (question, answer) tuples to evaluate.
            limit: Optional limit on the number of pairs to evaluate.
            progress: Whether to show a progress bar.
            stop_on_error: Whether to stop evaluation on the first error.
            on_result: Checkpoint each attempted result before starting another request.
            progress_initial: Valid saved scores already completed for this run.
            progress_total: Full run size, including saved scores.

        Otherwise, retain attempted results and stop on a permanent error or
        three consecutive failures so callers can save and resume partial work.
        """
        out: list[dict] = []
        completed = progress_initial
        # Determine total length for progress bar
        total: Optional[int] = None
        try:
            total = len(pairs)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            total = None
        if limit is not None:
            if total is None:
                total = limit
            else:
                total = min(total, limit)
        total = progress_total if progress_total is not None else (
            total + progress_initial if total is not None else None
        )
        bar = None
        if progress:
            try:
                bar = tqdm(total=total, initial=progress_initial, desc="Evaluating", unit="qa")
            except Exception:  # noqa: BLE001
                bar = None
        emit_progress("judging", progress_initial, total)
        consecutive_errors = 0
        for i, (q, a) in enumerate(pairs):
            if limit is not None and i >= limit:
                break
            should_stop = False
            try:
                res = self.evaluate(q, a)
                if not has_rubric_scores(res):
                    raise ValueError("Judge returned no evaluation scores")
                out.append({"index": i, "question": q, "answer": a, "evaluation": res})
                consecutive_errors = 0
            except Exception as e:  # noqa: BLE001
                if stop_on_error:
                    if bar:
                        bar.close()
                    raise
                out.append({"index": i, "question": q, "error": str(e)})
                consecutive_errors += 1
                tqdm.write(f"[error] Evaluation {i + 1}: {e}")
                if _stop_after_error(e, consecutive_errors):
                    tqdm.write("[failed] Stopping judge batch; returning attempted results for resume.")
                    should_stop = True
            # Keep persistence outside the provider error handler: a disk failure
            # must stop immediately, not trigger more paid requests.
            if on_result is not None:
                on_result(out[-1])
            if "error" not in out[-1]:
                completed += 1
                if bar:
                    bar.update(1)
            emit_progress("judging", completed, total)
            if should_stop:
                break
        if bar:
            bar.close()
        return out

    # -------------- Dataset convenience --------------
    @staticmethod
    def load_qa_pairs(jsonl_path: str) -> List[Tuple[str, str]]:
        return load_qa_pairs(jsonl_path)

    def evaluate_dataset(
        self,
        jsonl_path: str,
        limit: Optional[int] = None,
        progress: bool = True,
    ) -> dict:
        """Load a dataset and batch evaluate.

        Returns dict with raw results list and a small summary aggregate.
        """
        pairs = self.load_qa_pairs(jsonl_path)
        results = self.batch_evaluate(pairs, limit=limit, progress=progress)
        summary: Dict[str, Any] = {"total_evaluated": len(results)}
        if self.language == "arabic":
            purity_counts: Dict[int, int] = {}
            for r in results:
                eval_section = (
                    r.get("evaluation", {}).get("Arabic_Accuracy")
                    if "evaluation" in r
                    else None
                )
                if eval_section:
                    p = eval_section.get("Arabic_Purity")
                    if isinstance(p, int):
                        purity_counts[p] = purity_counts.get(p, 0) + 1
            summary["arabic_purity_distribution"] = purity_counts
        return {"results": results, "summary": summary}

    # -------------- Unified response generation --------------
    def generate_responses(
        self,
        questions: List[str],
        provider: str = "openai",
        model: Optional[str] = None,
        progress: bool = True,
        system: Optional[str] = None,
        on_result: Optional[Callable[[dict], None]] = None,
        progress_initial: int = 0,
        progress_total: Optional[int] = None,
        **kwargs,
    ) -> GenerationBatch:
        """Generate responses using any supported provider.

        Args:
            questions: List of questions to generate responses for
            provider: One of 'openai', 'together', 'hf', 'gemini', 'grok'
            model: Model name to use (if None, uses provider default)
            progress: Whether to show progress bar
            system: Optional system prompt to pass to provider (None -> no system message)
            **kwargs: Additional arguments passed to the provider wrapper

        Returns attempted responses. If ``aborted`` is true, persist successful
        answers and defer further generation and judging to a later invocation.
        """
        # Provider mapping
        provider_classes = {
            "openai": ParrotAIOpenAI,
            "together": ParrotAITogether,
            "hf": ParrotAIHF,
            "google": ParrotAIGemini,
            "xai": ParrotAIGrok,
            "anthropic": ParrotAIClaude,
        }

        # Default models per provider
        default_models = {
            "openai": "gpt-5-mini",
            "together": "Qwen/Qwen3.5-9B",
            "hf": "google/gemma-3-27b-it",
            "google": "gemini-3.8-flash",
            "xai": "grok-3-mini",
            "anthropic": "claude-haiku-4-5-20251001",
        }

        if provider not in provider_classes:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be one of: {list(provider_classes.keys())}"
            )

        # Initialize the appropriate wrapper
        wrapper_class = provider_classes[provider]
        if provider == "hf":
            # HF wrapper needs provider argument
            hf_provider = kwargs.get("hf_provider", "nebius")
            wrapper = wrapper_class(language=self.language, provider=hf_provider)
        else:
            wrapper = wrapper_class(language=self.language)

        # Set model if provided
        use_model = model or default_models[provider]
        if model:
            wrapper.set_model(model)

        out = GenerationBatch()
        total = progress_total if progress_total is not None else len(questions) + progress_initial
        bar = None
        if progress:
            try:
                bar = tqdm(
                    total=total,
                    initial=progress_initial,
                    desc=f"Generating ({provider.title()})",
                    unit="q",
                )
            except Exception:
                bar = None

        emit_progress("generating", progress_initial, total)
        completed = progress_initial
        consecutive_errors = 0
        for i, q in enumerate(questions):
            should_stop = False
            try:
                answer = wrapper.generate(prompt=q, model=use_model, system=system)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Provider returned an empty answer")
                out.append(
                    {
                        "index": i,
                        "question": q,
                        "answer": answer,
                        "model": use_model,
                        "provider": provider,
                        "system_used": bool(system),
                    }
                )
                consecutive_errors = 0
            except TerminalGenerationFailure as e:
                out.append({
                    "index": i, "question": q, "answer": "", "model": use_model,
                    "provider": provider, "system_used": bool(system),
                    "generation_failure": e.metadata(),
                })
                consecutive_errors = 0
                tqdm.write(f"[model failure] Generation {i + 1}: {e.reason}; recorded for minimum scoring, no retry")
            except Exception as e:
                out.append(
                    {
                        "index": i,
                        "question": q,
                        "error": str(e),
                        "model": use_model,
                        "provider": provider,
                        "system_used": bool(system),
                    }
                )
                consecutive_errors += 1
                tqdm.write(f"[error] Generation {i + 1}: {e}")
                if _stop_after_error(e, consecutive_errors):
                    out.aborted = True
                    tqdm.write("[failed] Stopping generation batch; returning attempted answers for resume.")
                    should_stop = True

            if on_result is not None:
                on_result(out[-1])
            if "error" not in out[-1]:
                completed += 1
                if bar:
                    bar.update(1)
            emit_progress("generating", completed, total)
            if should_stop:
                break

        if bar:
            bar.close()
        return out

    def generate_responses_from_file(
        self,
        question_file: str = "data/arabic/ar_eval_questions.txt",
        limit: int = 100,
        **kwargs,
    ) -> GenerationBatch:
        """Load questions from file then call ``generate_responses``."""
        questions = load_eval_questions(question_file, limit=limit)
        return self.generate_responses(questions, **kwargs)


__all__ = [
    "EvaluationEngine",
    "GenerationBatch",
    "load_qa_pairs",
    "load_eval_questions",
    "EvaluationResultArabic",
    "EvaluationResultEnglish",
    "has_scripture_citation",
    "has_theological_terminology",
    "calibrate_english_scores",
    "has_arabic_scripture_citation",
    "has_arabic_theological_terminology",
    "calibrate_arabic_scores",
]
