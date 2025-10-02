"""Sermon evaluation engine (language-agnostic):

Provides two-step evaluation per resources/Sermon Evaluation Framework.md:
 - Step 1: Structural extraction into JSON (deterministic)
 - Step 2: Analytical scoring with 1–5 rubric

Supports providers:
 - Text-only: openai, gemini
 - Audio (mp3, wav, m4a): gemini only, via Files API. Caches upload mapping.

CLI wrapper is implemented separately in cp_eval_sermons.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import itertools
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from .evaluation_schemas import (
    SermonExtractionStep1,
    SermonScoringStep2,
    SermonScoringStep2Raw,
    AggregatedSummary,
    AggregatedSummaryFeedback,
)
from .core import ParrotAIOpenAI, ParrotAIGemini

load_dotenv()


def _load_sermon_prompts():
    from .prompts import sermon as prompts
    return prompts


def _provider_from_name(name: str):
    name = name.lower()
    if name == "openai":
        return ParrotAIOpenAI(language="english")
    if name == "gemini":
        return ParrotAIGemini(language="english")
    raise ValueError(f"Unsupported provider: {name}")


class SermonEvaluationEngine:
    """Two-step sermon evaluation runner.

    Note: Language-agnostic; assumes English prompts. For other languages, add a new
    prompts module or extend routing.
    """

    def __init__(self, provider: str = "gemini", model: str = "gemini-2.5-flash") -> None:
        self.provider_name = provider
        self.model = model
        self.prompts = _load_sermon_prompts()
        self.provider = _provider_from_name(provider)
        self.provider.set_model(model)
        # Audio cache file: maps local path -> remote file id/url.
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.audio_cache_path = self.cache_dir / "gemini_files_cache.json"
        if not self.audio_cache_path.exists():
            self.audio_cache_path.write_text("{}", encoding="utf-8")

    # ----------- UX helpers -----------
    @staticmethod
    def _fmt_size(bytes_count: int) -> str:
        try:
            mb = bytes_count / (1024 * 1024)
            if mb >= 1024:
                gb = mb / 1024
                return f"{gb:.2f} GB"
            return f"{mb:.2f} MB"
        except Exception:
            return f"{bytes_count} B"

    @staticmethod
    @contextmanager
    def _upload_indicator(message: str = "Working"):
        """ASCII spinner shown while a blocking operation runs (no tqdm dependency)."""
        stop = threading.Event()

        def spin():
            for ch in itertools.cycle("|/-\\"):
                if stop.is_set():
                    break
                print(f"\r{message} {ch}", end="", flush=True)
                time.sleep(0.1)
            # clear line
            print("\r", end="")

        t = threading.Thread(target=spin, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=0.2)
            print(f"{message} done.")

    # ----------- Step 1: Extraction -----------
    def extract_structure_from_text(self, transcript: str) -> SermonExtractionStep1:
        prompt = (
            f"Instructions:\n{self.prompts.EXTRACTION_INSTRUCTIONS_TEXT}\n\n"
            "Return ONLY JSON. Transcript below:\n\n" + transcript
        )
        system = self.prompts.EXTRACTION_SYSTEM_PROMPT
        # Use structured parsing if available (OpenAI/Gemini), else best-effort text JSON parse
        with self._upload_indicator(message="Step 1: Extracting structure from transcript"):
            if isinstance(self.provider, ParrotAIGemini):
                data = self.provider.generate_structured(
                    prompt=f"{self.prompts.EXTRACTION_INSTRUCTIONS_TEXT}\n\n{transcript}",
                    response_schema=SermonExtractionStep1,
                    system=system,
                    model=self.model,
                )
            elif isinstance(self.provider, ParrotAIOpenAI):
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": self.prompts.EXTRACTION_INSTRUCTIONS_TEXT},
                    {"role": "user", "content": transcript},
                ]
                data = self.provider.generate_structured(
                    messages=msgs,
                    response_model=SermonExtractionStep1,
                    model=self.model
                )
            else:
                raw = self.provider.generate(prompt, system=system, model=self.model)
                data = self._safe_json_parse(raw)
        return SermonExtractionStep1(**data)

    # ----------- Step 2: Scoring -----------
    def score_from_extraction(self, extraction: SermonExtractionStep1) -> SermonScoringStep2:
        extraction_json = json.dumps(extraction.model_dump(), ensure_ascii=False)
        scoring_prompt = (
            f"{self.prompts.SCORING_INSTRUCTIONS}\n\n"
            f"Step 1 JSON below:\n\n{extraction_json}"
        )
        system = self.prompts.SCORING_SYSTEM_PROMPT
        with self._upload_indicator(message="Step 2: Scoring sermon (rubric)"):
            if isinstance(self.provider, ParrotAIGemini):
                data = self.provider.generate_structured(
                    prompt=scoring_prompt,
                    response_schema=SermonScoringStep2Raw,
                    system=system,
                    model=self.model,
                )
            elif isinstance(self.provider, ParrotAIOpenAI):
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": scoring_prompt},
                ]
                data = self.provider.generate_structured(
                    messages=msgs,
                    response_model=SermonScoringStep2Raw,
                    model=self.model
                )
            else:
                raw = self.provider.generate(scoring_prompt, system=system, model=self.model)
                data = self._safe_json_parse(raw)
        raw_scoring = SermonScoringStep2Raw(**data)
        
        # Convert raw to full scoring object
        scoring = SermonScoringStep2(
            Introduction=raw_scoring.Introduction,
            Proposition=raw_scoring.Proposition,
            Main_Points=raw_scoring.Main_Points,
            Exegetical_Support=raw_scoring.Exegetical_Support,
            Application=raw_scoring.Application,
            Illustrations=raw_scoring.Illustrations,
            Conclusion=raw_scoring.Conclusion,
            Strengths=raw_scoring.Strengths,
            Growth_Areas=raw_scoring.Growth_Areas,
            Next_Steps=raw_scoring.Next_Steps,
            Scoring_Confidence=raw_scoring.Scoring_Confidence,
        )
        # Compute aggregated summary per framework and attach
        scoring.Aggregated_Summary = self._compute_aggregates(scoring, extraction)

        # Generate feedback and explanations for the aggregate scores
        step2_dump = scoring.model_dump(exclude={"Aggregated_Summary_Feedback"})
        agg_prompt = (
            f"{self.prompts.AGG_SUMMARY_INSTRUCTIONS}\n\n"
            f"Step 1 JSON:\n{extraction_json}\n\n"
            f"Step 2 JSON:\n{json.dumps(step2_dump, ensure_ascii=False)}\n\n"
            f"Aggregated Summary JSON:\n"
            f"{json.dumps(scoring.Aggregated_Summary.model_dump() if scoring.Aggregated_Summary else {}, ensure_ascii=False)}"
        )

        agg_feedback_data = None
        try:
            with self._upload_indicator(message="Summarizing aggregate insights"):
                if isinstance(self.provider, ParrotAIGemini):
                    agg_feedback_data = self.provider.generate_structured(
                        prompt=agg_prompt,
                        response_schema=AggregatedSummaryFeedback,
                        system=self.prompts.AGG_SUMMARY_SYSTEM_PROMPT,
                        model=self.model,
                    )
                elif isinstance(self.provider, ParrotAIOpenAI):
                    msgs = [
                        {"role": "system", "content": self.prompts.AGG_SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": agg_prompt},
                    ]
                    agg_feedback_data = self.provider.generate_structured(
                        messages=msgs,
                        response_model=AggregatedSummaryFeedback,
                        model=self.model,
                    )
                else:
                    raw = self.provider.generate(
                        agg_prompt,
                        system=self.prompts.AGG_SUMMARY_SYSTEM_PROMPT,
                        model=self.model,
                    )
                    agg_feedback_data = self._safe_json_parse(raw)
        except Exception as exc:
            print(f"[sermons] Warning: could not generate aggregate feedback: {exc}")

        if agg_feedback_data:
            try:
                scoring.Aggregated_Summary_Feedback = AggregatedSummaryFeedback(**agg_feedback_data)
            except Exception as exc:
                print(f"[sermons] Warning: could not parse aggregate feedback JSON: {exc}")

        return scoring

    # ----------- Aggregates computation -----------
    @staticmethod
    def _clamp(v: float, lo: float = 1.0, hi: float = 5.0) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _avg(vals):
        lst = [v for v in vals if v is not None]
        return sum(lst) / len(lst) if lst else 1.0

    def _compute_aggregates(self, scoring: SermonScoringStep2, extraction: SermonExtractionStep1) -> AggregatedSummary:
        # Component rollups (all on 1–5)
        textual_fidelity = self._avg([
            scoring.Exegetical_Support.Alignment_with_Text,
            scoring.Exegetical_Support.Handles_Difficulties,
            scoring.Exegetical_Support.Proof_Accuracy_and_Clarity,
            scoring.Exegetical_Support.Context_and_Genre_Considered,
        ])

        proposition_clarity = self._avg([
            scoring.Proposition.Principle_and_Application_Wed,
            scoring.Proposition.Establishes_Main_Theme,
            scoring.Proposition.Summarizes_Introduction,
        ])

        fcf_identification = float(scoring.Introduction.FCF_Introduced)

        application_effectiveness = self._avg([
            scoring.Application.Clear_and_Practical,
            scoring.Application.Redemptive_Focus,
            scoring.Application.Mandate_vs_Idea_Distinction,
            scoring.Application.Passage_Supported,
            scoring.Main_Points.Application_Quality,
        ])

        structure_cohesion = self._avg([
            scoring.Main_Points.Proportional_and_Coexistent,
            scoring.Conclusion.Summary,
            scoring.Conclusion.Compelling_Exhortation,
            scoring.Conclusion.Climax,
            scoring.Conclusion.Pointed_End,
        ])

        illustrations = self._avg([
            scoring.Main_Points.Illustration_Quality,
            scoring.Illustrations.Lived_Body_Detail,
            scoring.Illustrations.Strengthens_Points,
            scoring.Illustrations.Proportion,
        ])

        # Weighted Overall Impact
        overall_base = (
            0.30 * textual_fidelity
            + 0.20 * proposition_clarity
            + 0.15 * application_effectiveness
            + 0.15 * structure_cohesion
            + 0.10 * illustrations
            + 0.10 * fcf_identification
        )
        overall = self._clamp(overall_base)

        # Round to two decimals per framework guidance
        def r2(x: float) -> float:
            return float(f"{x:.2f}")

        return AggregatedSummary(
            Textual_Fidelity=r2(textual_fidelity),
            Proposition_Clarity=r2(proposition_clarity),
            FCF_Identification=r2(fcf_identification),
            Application_Effectiveness=r2(application_effectiveness),
            Structure_Cohesion=r2(structure_cohesion),
            Illustrations=r2(illustrations),
            Overall_Impact_Base=r2(overall_base),
            Overall_Impact=r2(overall),
        )

    # ----------- Audio support (Gemini only) -----------
    def _load_audio_cache(self) -> Dict[str, Any]:
        try:
            return json.loads(self.audio_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_audio_cache(self, cache: Dict[str, Any]) -> None:
        self.audio_cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def upload_or_get_gemini_file(self, local_path: str) -> Tuple[str, Any]:
        """Return (remote_id, file_obj). Caches by absolute path.

        Requires provider == gemini. Uses google-genai files API.
        """
        if self.provider_name != "gemini":
            raise ValueError("Audio is only supported with provider=gemini")
        abs_path = str(Path(local_path).expanduser().resolve())
        cache = self._load_audio_cache()
        if abs_path in cache:
            remote_id = cache[abs_path]
            print(f"[sermons] [cache] Using previously uploaded file:\n  {abs_path} -> {remote_id}")
            # Try fetching a live file object so contents include a valid file reference
            file_obj = None
            try:
                if hasattr(self.provider, "get_file"):
                    file_obj = self.provider.get_file(remote_id)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[sermons] [cache] Could not fetch file object for {remote_id}: {e}")
                # As a fallback, attempt to re-upload to get a fresh object
                try:
                    with self._upload_indicator(message="Re-uploading to Gemini"):
                        file_obj = self.provider.upload_file(abs_path)  # type: ignore[attr-defined]
                    # Update cache mapping with the canonical name/id
                    cache[abs_path] = (
                        getattr(file_obj, "name", None)
                        or getattr(file_obj, "uri", None)
                        or getattr(file_obj, "id", None)
                    )
                    remote_id = cache[abs_path]
                    self._save_audio_cache(cache)
                    print(f"[sermons] [upload] Re-uploaded successfully -> {remote_id}")
                except Exception as ee:
                    print(f"[sermons] [upload] Re-upload failed: {ee}")
            return remote_id, file_obj
        # Use provider's helper
        try:
            size_bytes = os.path.getsize(abs_path)
        except Exception:
            size_bytes = -1
        size_str = self._fmt_size(size_bytes) if size_bytes >= 0 else "unknown size"
        print(f"[sermons] [upload] Uploading to Gemini Files API:\n  {abs_path} ({size_str})")
        with self._upload_indicator(message="Uploading to Gemini"):
            file_obj = self.provider.upload_file(abs_path)  # type: ignore[attr-defined]
        cache[abs_path] = (
            getattr(file_obj, "name", None)
            or getattr(file_obj, "uri", None)
            or getattr(file_obj, "id", None)
        )
        self._save_audio_cache(cache)
        print(f"[sermons] [upload] Uploaded successfully -> {cache[abs_path]}")
        return cache[abs_path], file_obj

    def extract_structure_from_audio(self, audio_path: str) -> SermonExtractionStep1:
        if self.provider_name != "gemini":
            raise ValueError("Audio extraction currently supported only for Gemini")
        file_id, file_obj = self.upload_or_get_gemini_file(audio_path)
        # Some cached entries may only return the id; try to fetch the object now
        if file_obj is None and hasattr(self.provider, "get_file"):
            try:
                file_obj = self.provider.get_file(file_id)  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[sermons] Warning: could not retrieve file object for {file_id}; proceeding with id only. {e}")
        # Construct Gemini content: instruction + file
        instruction = self.prompts.EXTRACTION_INSTRUCTIONS_AUDIO
        system = self.prompts.EXTRACTION_SYSTEM_PROMPT
        # Use two-part content: instruction and file reference
        with self._upload_indicator(message="Step 1: Extracting structure from audio"):
            if isinstance(self.provider, ParrotAIGemini):
                data = self.provider.generate_structured_with_contents(
                    contents=[instruction, file_obj or file_id],
                    response_schema=SermonExtractionStep1,
                    system=system,
                    model=self.model,
                )
            else:
                raw = self.provider.generate_with_contents([instruction, file_obj or file_id], system=system, model=self.model)  # type: ignore[attr-defined]
                data = self._safe_json_parse(raw)
        return SermonExtractionStep1(**data)

    # ----------- Helpers -----------
    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        text = text.strip()
        # Some providers may wrap JSON in code fences
        if text.startswith("```"):
            text = text.strip('`')
            if text.lower().startswith("json"):
                text = text[4:]  # remove 'json' label
        try:
            return json.loads(text)
        except Exception:
            # Last resort: attempt to locate first and last braces
            first = text.find('{')
            last = text.rfind('}')
            if first != -1 and last != -1 and last > first:
                return json.loads(text[first:last+1])
            raise


__all__ = [
    "SermonEvaluationEngine",
]
