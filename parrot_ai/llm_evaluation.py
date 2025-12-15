"""Evaluation utilities and engine class for Arabic theological QA model assessment.

This module exposes an ``EvaluationEngine`` class encapsulating the earlier
function-based workflow so notebooks can simply do:

    from parrot_ai.evaluation import EvaluationEngine, default_engine
    engine = EvaluationEngine(model="gpt-5-mini")  # or use default_engine
    pairs = engine.load_qa_pairs("data/arabic/ar_training_dataset_small_model.jsonl")
    results = engine.batch_evaluate(pairs, limit=10)
"""

import os
import json
import importlib
from typing import List, Tuple, Dict, Any, Optional, Iterable, cast
from openai import OpenAI
from google import genai
from google.genai import types
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
)

load_dotenv()

DEFAULT_MODEL = "gpt-5-mini"


# --- Data loading utility ---
def load_qa_pairs(
    jsonl_path: str,
    question_list_path: Optional[str] = "data/arabic/ar_eval_questions.txt",
    limit: int = 100,
) -> List[Tuple[str, str]]:
    """Load (question, answer) pairs filtered to a curated eval question list.

    Behavior change (per user request): Instead of loading every pair in the
    dataset, we now restrict to the first ``limit`` questions present in the
    ``question_list_path`` file (one question per line) if that file exists.

    If ``question_list_path`` is None or the file is missing/empty, the
    function falls back to previous behavior (load all pairs) but still caps
    at ``limit`` if provided (>0).
    """
    question_filter: Optional[List[str]] = None
    question_set: Optional[set[str]] = None
    if question_list_path and limit != 0:
        try:
            with open(question_list_path, "r", encoding="utf-8") as qf:
                question_filter = [ln.strip() for ln in qf if ln.strip()][:limit]
            if question_filter:
                question_set = set(question_filter)
        except FileNotFoundError:
            question_filter = None
            question_set = None
    pairs: List[Tuple[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if question_set and len(pairs) >= len(question_set):
                break
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = obj.get("messages") or []
            if len(messages) >= 3:
                user = (
                    messages[1].get("content")
                    if isinstance(messages[1], dict)
                    else None
                )
                assistant = (
                    messages[2].get("content")
                    if isinstance(messages[2], dict)
                    else None
                )
                if user and assistant:
                    if question_set is None or user in question_set:
                        pairs.append((user, assistant))
            elif len(messages) == 2:
                user = (
                    messages[0].get("content")
                    if isinstance(messages[0], dict)
                    else None
                )
                assistant = (
                    messages[1].get("content")
                    if isinstance(messages[1], dict)
                    else None
                )
                if user and assistant:
                    if question_set is None or user in question_set:
                        pairs.append((user, assistant))
    # Re-order according to the question list if present
    if question_filter and pairs:
        order_map = {q: i for i, q in enumerate(question_filter)}
        pairs.sort(key=lambda qa: order_map.get(qa[0], 10_000_000))
    # Apply limit fallback if no question list was used
    if question_set is None and limit and limit > 0:
        pairs = pairs[:limit]
    return pairs


def load_eval_questions(
    question_file: str = "data/arabic/ar_eval_questions.txt", limit: Optional[int] = 100
) -> List[str]:
    """Load up to ``limit`` evaluation questions from text file (one per line).

    If limit is None or 0, load all questions.
    """
    try:
        with open(question_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            if limit and limit > 0:
                return lines[:limit]
            return lines
    except FileNotFoundError:
        return []


ARABIC_BLOCKS = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0x0870, 0x089F),  # Arabic Extended-B
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0x1EE00, 0x1EEFF),  # Arabic Mathematical Alphabetic Symbols
]


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in ARABIC_BLOCKS)


def basic_language_metrics(text: str) -> Dict[str, Any]:
    """Compute simple Arabic vs non-Arabic letter percentages."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"arabic_char_pct": 0.0, "non_arabic_char_pct": 0.0, "total_letters": 0}
    arabic = sum(1 for c in letters if is_arabic_char(c))
    arabic_pct = arabic / len(letters) * 100
    return {
        "arabic_char_pct": round(arabic_pct, 2),
        "non_arabic_char_pct": round(100 - arabic_pct, 2),
        "total_letters": len(letters),
    }


# Post-processing: enforce purity & grammar caps based on heuristic percentage
def apply_purity_penalty(answer: str, result_dict: dict) -> dict:
    """Apply heuristic purity cap and related grammar adjustments."""
    lang_metrics = basic_language_metrics(answer)
    purity_pct = lang_metrics["arabic_char_pct"]
    if purity_pct >= 98:
        cap = 5
    elif purity_pct >= 90:
        cap = 4
    elif purity_pct >= 75:
        cap = 3
    elif purity_pct >= 60:
        cap = 2
    else:
        cap = 1
    arabic_section = result_dict.get("Arabic_Accuracy", {})
    if arabic_section.get("Arabic_Purity", cap) > cap:
        arabic_section["Arabic_Purity"] = cap
        reason = arabic_section.get("Penalty_Reason") or ""
        if reason:
            reason += " | "
        arabic_section["Penalty_Reason"] = (
            reason + f"Capped purity (heuristic {purity_pct}%)"
        )
    if cap <= 2:
        if arabic_section.get("Grammar_and_Syntax", 5) > 3:
            arabic_section["Grammar_and_Syntax"] = 3
            reason = arabic_section.get("Penalty_Reason") or ""
            if reason:
                reason += " | "
            arabic_section["Penalty_Reason"] = (
                reason + "Grammar capped due to low purity"
            )
        if arabic_section.get("Overall", 5) > 3:
            arabic_section["Overall"] = min(3, arabic_section["Overall"])
    arabic_section["Heuristic_Arabic_Purity_Pct"] = purity_pct
    result_dict["Arabic_Accuracy"] = arabic_section
    return result_dict


# Clamp Overall within ±1 of rounded mean of its component keys (model honesty guardrail)
def clamp_overall(section: dict, keys: list[str]) -> None:
    vals = [section[k] for k in keys if isinstance(section.get(k), int)]
    if not vals or "Overall" not in section or not isinstance(section["Overall"], int):
        return
    target = round(sum(vals) / len(vals))
    low, high = target - 1, target + 1
    section["Overall"] = min(max(section["Overall"], low), high)


def clamp_all_overalls(result_dict: dict) -> None:
    clamp_overall(
        result_dict.get("Adherence", {}),
        ["Core", "Secondary", "Tertiary_Handling", "Biblical_Basis", "Consistency"],
    )
    clamp_overall(
        result_dict.get("Kindness_and_Gentleness", {}),
        [
            "Core_Clarity_with_Kindness",
            "Pastoral_Sensitivity",
            "Secondary_Fairness",
            "Tertiary_Neutrality",
            "Tone",
        ],
    )
    clamp_overall(
        result_dict.get("Interfaith_Sensitivity", {}),
        [
            "Respect_and_Handling_Objections",
            "Objection_Acknowledgement",
            "Evangelism",
            "Gospel_Boldness",
        ],
    )
    arabic_section = result_dict.get("Arabic_Accuracy")
    if isinstance(arabic_section, dict):  # Only for Arabic runs
        clamp_overall(
            arabic_section,
            [
                "Grammar_and_Syntax",
                "Theological_Nuance",
                "Contextual_Clarity",
                "Consistency_of_Terms",
                "Arabic_Purity",
            ],
        )


# Clamp all scale scores to [1, 5]
def clamp_scale_scores(d: dict) -> dict:
    def clip(v):
        return max(1, min(5, int(v))) if isinstance(v, int) else 1

    for sect_key, sect in d.items():
        if not isinstance(sect, dict):
            continue
        for k, v in list(sect.items()):
            if k in ("Penalty_Reason", "Heuristic_Arabic_Purity_Pct"):
                continue
            sect[k] = clip(v)
    return d


def enforce_knockouts(answer: str, result_dict: dict) -> dict:
    """Apply rubric knockout rules and empty-answer handling."""
    if not answer.strip():
        # Only create base sections; Arabic_Accuracy is added elsewhere (engine) for Arabic language
        for section_key, fields in [
            (
                "Adherence",
                [
                    "Core",
                    "Secondary",
                    "Tertiary_Handling",
                    "Biblical_Basis",
                    "Consistency",
                    "Overall",
                ],
            ),
            (
                "Kindness_and_Gentleness",
                [
                    "Core_Clarity_with_Kindness",
                    "Pastoral_Sensitivity",
                    "Secondary_Fairness",
                    "Tertiary_Neutrality",
                    "Tone",
                    "Overall",
                ],
            ),
            (
                "Interfaith_Sensitivity",
                [
                    "Respect_and_Handling_Objections",
                    "Objection_Acknowledgement",
                    "Evangelism",
                    "Gospel_Boldness",
                    "Overall",
                ],
            ),
        ]:
            section = result_dict.get(section_key, {})
            for f in fields:
                section[f] = 1
            result_dict[section_key] = section
        if "Arabic_Accuracy" in result_dict:
            arabic_section = result_dict["Arabic_Accuracy"]
            for f in [
                "Grammar_and_Syntax",
                "Theological_Nuance",
                "Contextual_Clarity",
                "Consistency_of_Terms",
                "Arabic_Purity",
                "Overall",
            ]:
                arabic_section[f] = 1
            arabic_section["Penalty_Reason"] = "Empty answer"
            result_dict["Arabic_Accuracy"] = arabic_section
        return result_dict

    adherence = result_dict.get("Adherence", {})
    if (
        isinstance(adherence.get("Core"), int)
        and adherence.get("Core", 5) <= 2
        and adherence.get("Overall", 5) > 3
    ):
        adherence["Overall"] = 3
    result_dict["Adherence"] = adherence

    interfaith = result_dict.get("Interfaith_Sensitivity", {})
    if (
        isinstance(interfaith.get("Respect_and_Handling_Objections"), int)
        and interfaith.get("Respect_and_Handling_Objections", 5) <= 1
        and interfaith.get("Overall", 5) > 2
    ):
        interfaith["Overall"] = 2
    result_dict["Interfaith_Sensitivity"] = interfaith

    arabic = result_dict.get("Arabic_Accuracy", {})
    if (
        isinstance(arabic.get("Arabic_Purity"), int)
        and arabic.get("Arabic_Purity", 5) <= 2
        and arabic.get("Grammar_and_Syntax", 5) > 3
    ):
        arabic["Grammar_and_Syntax"] = 3
        reason = arabic.get("Penalty_Reason") or ""
        if reason:
            reason += " | "
        arabic["Penalty_Reason"] = (
            reason + "Grammar capped due to low purity (knockout)"
        )
    result_dict["Arabic_Accuracy"] = arabic
    return result_dict


# --- Boldness / anti-relativism heuristic adjustments ---
def adjust_boldness(
    answer: str,
    result_dict: dict,
    bold_keywords: list[str],
    relativism_patterns: list[str],
) -> dict:
    interfaith = result_dict.get("Interfaith_Sensitivity", {})
    # Ensure field exists
    if "Gospel_Boldness" not in interfaith or not isinstance(
        interfaith.get("Gospel_Boldness"), int
    ):
        interfaith["Gospel_Boldness"] = 3
    lower_ans = answer.lower()
    has_relativism = any(pat.lower() in lower_ans for pat in relativism_patterns)
    has_bold = any(kw.lower() in lower_ans for kw in bold_keywords)
    # Penalize relativism if no bold Christ-centered content
    if has_relativism and not has_bold:
        interfaith["Gospel_Boldness"] = min(interfaith.get("Gospel_Boldness", 3), 2)
        # Also cap Evangelism
        if interfaith.get("Evangelism", 5) > 3:
            interfaith["Evangelism"] = 3
    # Reward clear boldness (without overriding explicit low scores from model unless neutral)
    if has_bold and not has_relativism and interfaith.get("Gospel_Boldness", 0) < 4:
        interfaith["Gospel_Boldness"] = 4
    # If both strong bold keywords and explicit invitation words, consider 5
    if (
        has_bold
        and ("توب" in answer or "تعال" in answer or "آمن" in answer)
        and not has_relativism
    ):
        interfaith["Gospel_Boldness"] = max(interfaith["Gospel_Boldness"], 5)
    result_dict["Interfaith_Sensitivity"] = interfaith
    return result_dict


# Primary evaluation call


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
    ) -> None:
        """Create a language-aware evaluation engine.

        language: selects prompt module parrot_ai.prompts.<language>
        The module must define EVAL_SYSTEM_PROMPT and EVAL_INSTRUCTIONS.
        """
        self.language = language
        self.seed = seed
        self.model = model
        if model == DEFAULT_MODEL:
            self.client = client or OpenAI()
        else:
            if model.startswith("gemini"):
                self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
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
        # Optional heuristics
        self.relativism_patterns = getattr(self.prompts, "RELATIVISM_PATTERNS", [])
        self.bold_keywords = getattr(self.prompts, "BOLD_KEYWORDS", [])

    # -------------- Core single evaluation --------------
    def evaluate(self, question: str, answer: str) -> dict:
        """Evaluate a single (question, answer) pair returning rubric dict."""
        # Fast path for empty / whitespace-only answers: assign all 1s (lowest rubric) + penalty
        if not answer.strip():
            temp: dict[str, dict[str, int | str]] = {
                "Adherence": {
                    k: 1
                    for k in [
                        "Core",
                        "Secondary",
                        "Tertiary_Handling",
                        "Biblical_Basis",
                        "Consistency",
                        "Overall",
                    ]
                },
                "Kindness_and_Gentleness": {
                    k: 1
                    for k in [
                        "Core_Clarity_with_Kindness",
                        "Pastoral_Sensitivity",
                        "Secondary_Fairness",
                        "Tertiary_Neutrality",
                        "Tone",
                        "Overall",
                    ]
                },
                "Interfaith_Sensitivity": {
                    k: 1
                    for k in [
                        "Respect_and_Handling_Objections",
                        "Objection_Acknowledgement",
                        "Evangelism",
                        "Gospel_Boldness",
                        "Overall",
                    ]
                },
            }
            if self.language == "arabic":
                temp["Arabic_Accuracy"] = {
                    **{
                        k: 1
                        for k in [
                            "Grammar_and_Syntax",
                            "Theological_Nuance",
                            "Contextual_Clarity",
                            "Consistency_of_Terms",
                            "Arabic_Purity",
                            "Overall",
                        ]
                    },
                    "Penalty_Reason": "Empty answer",
                }
            return temp
        if self.language == "arabic":
            user_content = f"السؤال:\n{question}\n\nالإجابة:\n{answer}\n\nقيّم وفق التعليمات السابقة."
        else:
            user_content = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvaluate per the rubric instructions above."
        # Choose response model based on language
        response_model = (
            EvaluationResultArabic
            if self.language == "arabic"
            else EvaluationResultEnglish
        )
        if self.model.startswith("gemini"):
            # Gemini path
            cfg = types.GenerateContentConfig(
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
        result_dict = clamp_scale_scores(result_dict)
        # Only apply Arabic purity heuristics for Arabic language
        if self.language == "arabic" and "Arabic_Accuracy" in result_dict:
            result_dict = apply_purity_penalty(answer, result_dict)
        clamp_all_overalls(result_dict)
        result_dict = enforce_knockouts(answer, result_dict)
        result_dict = adjust_boldness(
            answer, result_dict, self.bold_keywords, self.relativism_patterns
        )
        clamp_all_overalls(result_dict)
        return result_dict

    # -------------- Batch utilities --------------
    def batch_evaluate(
        self,
        pairs: Iterable[Tuple[str, str]],
        limit: Optional[int] = None,
        progress: bool = True,
        stop_on_error: bool = False,
    ) -> list[dict]:
        """Evaluate multiple QA pairs.

        Args:
            pairs: An iterable of (question, answer) tuples to evaluate.
            limit: Optional limit on the number of pairs to evaluate.
            progress: Whether to show a progress bar.
            stop_on_error: Whether to stop evaluation on the first error.
        """
        out: list[dict] = []
        processed = 0
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
        bar = None
        if progress:
            try:
                bar = tqdm(total=total, desc="Evaluating", unit="qa")
            except Exception:  # noqa: BLE001
                bar = None
        for i, (q, a) in enumerate(pairs):
            if limit is not None and processed >= limit:
                break
            try:
                res = self.evaluate(q, a)
                out.append({"index": i, "question": q, "answer": a, "evaluation": res})
            except Exception as e:  # noqa: BLE001
                if stop_on_error:
                    if bar:
                        bar.close()
                    raise
                out.append({"index": i, "question": q, "error": str(e)})
            processed += 1
            if bar:
                bar.update(1)
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
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate responses using any supported provider.

        Args:
            questions: List of questions to generate responses for
            provider: One of 'openai', 'together', 'hf', 'gemini', 'grok'
            model: Model name to use (if None, uses provider default)
            progress: Whether to show progress bar
            system: Optional system prompt to pass to provider (None -> no system message)
            **kwargs: Additional arguments passed to the provider wrapper
        """
        # Provider mapping
        provider_classes = {
            "openai": ParrotAIOpenAI,
            "together": ParrotAITogether,
            "hf": ParrotAIHF,
            "google": ParrotAIGemini,
            "xai": ParrotAIGrok,
        }

        # Default models per provider
        default_models = {
            "openai": "gpt-5-mini",
            "together": "google/gemma-3-27b-it",
            "hf": "google/gemma-3-27b-it",
            "gemini": "gemini-2.5-flash",
            "grok": "grok-3-mini",
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

        out: List[Dict[str, Any]] = []
        bar = None
        if progress:
            try:
                bar = tqdm(
                    total=len(questions),
                    desc=f"Generating ({provider.title()})",
                    unit="q",
                )
            except Exception:
                bar = None

        for i, q in enumerate(questions):
            try:
                answer = wrapper.generate(prompt=q, model=use_model, system=system)
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

            if bar:
                bar.update(1)

        if bar:
            bar.close()
        return out

    def generate_responses_from_file(
        self,
        question_file: str = "data/arabic/ar_eval_questions.txt",
        limit: int = 100,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load questions from file then call ``generate_responses``."""
        questions = load_eval_questions(question_file, limit=limit)
        return self.generate_responses(questions, **kwargs)


__all__ = [
    "EvaluationEngine",
    "load_qa_pairs",
    "load_eval_questions",
    "EvaluationResultArabic",
    "EvaluationResultEnglish",
]
