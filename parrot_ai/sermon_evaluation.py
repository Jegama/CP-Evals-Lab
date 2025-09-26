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
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from .evaluation_schemas import (
    SermonExtractionStep1,
    SermonScoringStep2,
    SermonScoringStep2Raw,
    AggregatedSummary,
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

    # ----------- Step 1: Extraction -----------
    def extract_structure_from_text(self, transcript: str) -> SermonExtractionStep1:
        prompt = (
            f"Instructions:\n{self.prompts.EXTRACTION_INSTRUCTIONS_TEXT}\n\n"
            "Return ONLY JSON. Transcript below:\n\n" + transcript
        )
        system = self.prompts.EXTRACTION_SYSTEM_PROMPT
        # Use structured parsing if available (OpenAI/Gemini), else best-effort text JSON parse
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
        prompt = (
            f"Instructions:\n{self.prompts.SCORING_INSTRUCTIONS}\n\n"
            "Return ONLY JSON. Step 1 JSON below:\n\n" + extraction_json
        )
        system = self.prompts.SCORING_SYSTEM_PROMPT
        if isinstance(self.provider, ParrotAIGemini):
            data = self.provider.generate_structured(
                prompt=f"{self.prompts.SCORING_INSTRUCTIONS}\n\n{extraction_json}",
                response_schema=SermonScoringStep2Raw,
                system=system,
                model=self.model,
            )
        elif isinstance(self.provider, ParrotAIOpenAI):
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": self.prompts.SCORING_INSTRUCTIONS},
                {"role": "user", "content": extraction_json},
            ]
            data = self.provider.generate_structured(
                messages=msgs,
                response_model=SermonScoringStep2Raw,
                model=self.model
            )
        else:
            raw = self.provider.generate(prompt, system=system, model=self.model)
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

        # Optional narrative adjustment: not supplied programmatically; default 0
        adjustment = 0.0
        rationale = None

        overall = self._clamp(overall_base + adjustment)

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
            Overall_Impact_Adjustment=r2(adjustment),
            Adjustment_Rationale=rationale,
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
            return cache[abs_path], None
        # Use provider's helper
        file_obj = self.provider.upload_file(abs_path)  # type: ignore[attr-defined]
        cache[abs_path] = (
            getattr(file_obj, "name", None)
            or getattr(file_obj, "uri", None)
            or getattr(file_obj, "id", None)
        )
        self._save_audio_cache(cache)
        return cache[abs_path], file_obj

    def extract_structure_from_audio(self, audio_path: str) -> SermonExtractionStep1:
        if self.provider_name != "gemini":
            raise ValueError("Audio extraction currently supported only for Gemini")
        file_id, file_obj = self.upload_or_get_gemini_file(audio_path)
        # Construct Gemini content: instruction + file
        instruction = self.prompts.EXTRACTION_INSTRUCTIONS_AUDIO
        system = self.prompts.EXTRACTION_SYSTEM_PROMPT
        # Use two-part content: instruction and file reference
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
