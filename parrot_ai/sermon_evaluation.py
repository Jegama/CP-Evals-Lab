"""Sermon evaluation engine (language-agnostic):

Provides two-step evaluation per resources/Sermon Evaluation Framework.md:
 - Step 1: Structural extraction into JSON (deterministic)
 - Step 2: Analytical scoring with 1–5 rubric

Provider: Gemini only (supports both text and audio via Files API)
 - Text: transcript input
 - Audio: mp3, wav, m4a files via Files API with upload caching

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
from typing import Any, Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from .evaluation_schemas import (
    SermonExtractionStep1,
    SermonScoringStep2,
    SermonScoringStep2Raw,
    AggregatedSummary,
    AggregatedSummaryFeedback,
    IntroductionScores,
    PropositionScores,
    MainPointsScores,
    ExegeticalSupportScores,
    ApplicationScores,
    IllustrationsScores,
    ConclusionScores,
)

from .core import ParrotAIGemini

load_dotenv()


def _load_sermon_prompts():
    from .prompts import sermon as prompts

    return prompts


class SermonEvaluationEngine:
    """Two-step sermon evaluation runner.

    Note: Language-agnostic; assumes English prompts. For other languages, add a new
    prompts module or extend routing.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.provider_name = "gemini"
        self.model = model
        self.prompts = _load_sermon_prompts()
        self.provider = ParrotAIGemini(language="english")
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
    def get_audio_duration(file_path: str) -> Optional[float]:
        """Extract audio duration in seconds using ffprobe (preferred) or mutagen.

        Returns None if file cannot be parsed.
        """
        # 1. Try ffprobe first (most accurate for VBR/containers)
        try:
            import subprocess

            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            # Run with timeout to avoid hanging
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                val = result.stdout.strip()
                if val:
                    return float(val)
        except Exception:
            # ffprobe missing or failed, fall through to mutagen
            pass

        # 2. Fallback to mutagen
        try:
            from mutagen._file import File
            from mutagen.mp3 import MP3
            from mutagen.mp4 import MP4
            from mutagen.wave import WAVE
            from mutagen.flac import FLAC
            from mutagen.oggvorbis import OggVorbis

            # Try generic detection first
            audio = File(file_path)
            if audio is not None and audio.info:
                return float(audio.info.length)

            # Fallback: explicit detection by extension
            ext = Path(file_path).suffix.lower()
            if ext == ".mp3":
                audio = MP3(file_path)
            elif ext in [".m4a", ".mp4"]:
                audio = MP4(file_path)
            elif ext == ".wav":
                audio = WAVE(file_path)
            elif ext == ".flac":
                audio = FLAC(file_path)
            elif ext == ".ogg":
                audio = OggVorbis(file_path)

            if audio is not None and audio.info:
                return float(audio.info.length)

        except Exception as e:
            print(
                f"[sermons] Warning: could not extract audio duration from {file_path}: {e}"
            )
        return None

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
        system = self.prompts.EXTRACTION_SYSTEM_PROMPT
        with self._upload_indicator(
            message="Step 1: Extracting structure from transcript"
        ):
            data = self.provider.generate_structured(
                prompt=f"{self.prompts.EXTRACTION_INSTRUCTIONS_TEXT}\n\n{transcript}",
                response_schema=SermonExtractionStep1,
                system=system,
                model=self.model,
            )
        return SermonExtractionStep1(**data)

    # ----------- Step 2: Scoring -----------
    def score_from_extraction(
        self,
        extraction: SermonExtractionStep1,
        audio_file_obj: Optional[Any] = None,
    ) -> SermonScoringStep2:
        extraction_json = json.dumps(extraction.model_dump(), ensure_ascii=False)
        scoring_prompt = (
            f"{self.prompts.SCORING_INSTRUCTIONS}\\n\\n"
            f"Step 1 JSON below:\\n\\n{extraction_json}"
        )
        system = self.prompts.SCORING_SYSTEM_PROMPT
        with self._upload_indicator(message="Step 2: Scoring sermon (rubric)"):
            if audio_file_obj is not None:
                # Use multi-part content: instruction + audio for verification
                data = self.provider.generate_structured_with_contents(
                    contents=[scoring_prompt, audio_file_obj],
                    response_schema=SermonScoringStep2Raw,
                    system=system,
                    model=self.model,
                )
            else:
                # Text-only mode
                data = self.provider.generate_structured(
                    prompt=scoring_prompt,
                    response_schema=SermonScoringStep2Raw,
                    system=system,
                    model=self.model,
                )
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

        # Apply strict calibration heuristics to avoid inflated scores
        scoring = self._apply_strict_calibration(scoring, extraction)

        # Compute aggregated summary per framework and attach (after calibration)
        scoring.Aggregated_Summary = self._compute_aggregates(scoring, extraction)

        # Apply duration penalty if audio duration is available
        if extraction.audio_duration is not None:
            scoring.Aggregated_Summary = self._apply_duration_penalty(
                scoring.Aggregated_Summary, extraction.audio_duration
            )

        # Generate feedback and explanations for the aggregate scores
        step2_dump = scoring.model_dump(exclude={"Aggregated_Summary_Feedback"})

        # Include duration info in aggregate prompt if available
        duration_info = ""
        if extraction.audio_duration is not None:
            duration_minutes = extraction.audio_duration / 60.0
            duration_info = f"\\n\\nSermon Duration: {duration_minutes:.1f} minutes"
            if (
                scoring.Aggregated_Summary
                and scoring.Aggregated_Summary.duration_penalty
            ):
                duration_info += f" (penalty applied: {scoring.Aggregated_Summary.duration_penalty:.2f} points)"

        agg_prompt = (
            f"{self.prompts.AGG_SUMMARY_INSTRUCTIONS}\\n\\n"
            f"Step 1 JSON:\\n{extraction_json}\\n\\n"
            f"Step 2 JSON:\\n{json.dumps(step2_dump, ensure_ascii=False)}\\n\\n"
            f"Aggregated Summary JSON:\\n"
            f"{json.dumps(scoring.Aggregated_Summary.model_dump() if scoring.Aggregated_Summary else {}, ensure_ascii=False)}"
            f"{duration_info}"
        )

        agg_feedback_data = None
        try:
            with self._upload_indicator(message="Summarizing aggregate insights"):
                agg_feedback_data = self.provider.generate_structured(
                    prompt=agg_prompt,
                    response_schema=AggregatedSummaryFeedback,
                    system=self.prompts.AGG_SUMMARY_SYSTEM_PROMPT,
                    model=self.model,
                )
        except Exception as exc:
            print(f"[sermons] Warning: could not generate aggregate feedback: {exc}")

        if agg_feedback_data:
            try:
                scoring.Aggregated_Summary_Feedback = AggregatedSummaryFeedback(
                    **agg_feedback_data
                )
            except Exception as exc:
                print(
                    f"[sermons] Warning: could not parse aggregate feedback JSON: {exc}"
                )

        return scoring

    # ----------- Multi-run scoring for self-consistency -----------
    def _score_single_run(
        self,
        extraction: SermonExtractionStep1,
        audio_file_obj: Optional[Any],
        seed: int,
    ) -> Optional[SermonScoringStep2Raw]:
        """Execute a single Step 2 scoring run with specified seed.

        Returns None on failure (logs warning but does not raise).
        """
        try:
            extraction_json = json.dumps(extraction.model_dump(), ensure_ascii=False)
            scoring_prompt = (
                f"{self.prompts.SCORING_INSTRUCTIONS}\\n\\n"
                f"Step 1 JSON below:\\n\\n{extraction_json}"
            )
            system = self.prompts.SCORING_SYSTEM_PROMPT

            if audio_file_obj is not None:
                data = self.provider.generate_structured_with_contents(
                    contents=[scoring_prompt, audio_file_obj],
                    response_schema=SermonScoringStep2Raw,
                    system=system,
                    model=self.model,
                    seed=seed,
                )
            else:
                data = self.provider.generate_structured(
                    prompt=scoring_prompt,
                    response_schema=SermonScoringStep2Raw,
                    system=system,
                    model=self.model,
                    seed=seed,
                )
            return SermonScoringStep2Raw(**data)
        except Exception as exc:
            print(f"[sermons] Warning: scoring run with seed {seed} failed: {exc}")
            return None

    def score_from_extraction_multi_run(
        self,
        extraction: SermonExtractionStep1,
        audio_file_obj: Optional[Any],
        num_runs: int = 3,
    ) -> SermonScoringStep2:
        """Execute multiple parallel Step 2 scoring runs and harmonize results.

        Uses ThreadPoolExecutor for parallel API calls. Retries failed runs up to 2 batches
        until num_runs successful results are obtained. Returns harmonized scoring with
        averaged integers and synthesized feedback.
        """
        print(
            f"[sermons] Running {num_runs} parallel scoring runs for self-consistency..."
        )

        seeds = [1689, 2024, 3141, 4567, 5000, 6789, 7890, 8888, 9999]  # Pool of seeds
        runs: List[SermonScoringStep2Raw] = []
        seed_idx = 0
        max_retry_batches = 2
        retry_batch = 0

        while len(runs) < num_runs and retry_batch <= max_retry_batches:
            needed = num_runs - len(runs)
            batch_seeds = seeds[seed_idx : seed_idx + needed]
            seed_idx += needed

            if retry_batch > 0:
                print(
                    f"[sermons] Retry batch {retry_batch}: running {needed} more attempts..."
                )

            # Run in parallel
            with ThreadPoolExecutor(max_workers=min(needed, 5)) as executor:
                futures = {
                    executor.submit(
                        self._score_single_run, extraction, audio_file_obj, s
                    ): s
                    for s in batch_seeds
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        runs.append(result)

            retry_batch += 1

        if len(runs) < num_runs:
            print(
                f"[sermons] Warning: only {len(runs)}/{num_runs} runs succeeded. Proceeding with available results."
            )
        else:
            print(f"[sermons] Successfully completed {len(runs)} scoring runs.")

        if not runs:
            raise RuntimeError(
                "All scoring runs failed. Cannot proceed with multi-run evaluation."
            )

        # Harmonize (average + synthesize feedback)
        return self._harmonize_scoring_runs(runs, extraction, audio_file_obj)

    def _harmonize_scoring_runs(
        self,
        runs: List[SermonScoringStep2Raw],
        extraction: SermonExtractionStep1,
        audio_file_obj: Optional[Any],
    ) -> SermonScoringStep2:
        """Average numeric scores with confidence weighting and harmonize feedback via LLM.

        Implements self-consistency with confidence-based weighting:
        - Higher Scoring_Confidence runs contribute more to averaged scores
        - Harmonization LLM synthesizes feedback from all runs, noting consensus and dissent
        """
        print(f"[sermons] Harmonizing {len(runs)} scoring runs...")

        # Extract confidence scores
        confidences = [r.Scoring_Confidence for r in runs]
        total_conf = sum(confidences)
        weights = (
            [c / total_conf for c in confidences]
            if total_conf > 0
            else [1.0 / len(runs)] * len(runs)
        )

        def weighted_avg_int(values: List[int]) -> int:
            """Confidence-weighted average, rounded to nearest integer, clamped 1-5."""
            avg = sum(v * w for v, w in zip(values, weights))
            return max(1, min(5, round(avg)))

        # Average all integer rubric fields with confidence weighting

        averaged = SermonScoringStep2Raw(
            Introduction=IntroductionScores(
                FCF_Introduced=weighted_avg_int(
                    [r.Introduction.FCF_Introduced for r in runs]
                ),
                Arouses_Attention=weighted_avg_int(
                    [r.Introduction.Arouses_Attention for r in runs]
                ),
                Overall=weighted_avg_int([r.Introduction.Overall for r in runs]),
                Feedback="",  # Will be harmonized
            ),
            Proposition=PropositionScores(
                Principle_and_Application_Wed=weighted_avg_int(
                    [r.Proposition.Principle_and_Application_Wed for r in runs]
                ),
                Establishes_Main_Theme=weighted_avg_int(
                    [r.Proposition.Establishes_Main_Theme for r in runs]
                ),
                Summarizes_Introduction=weighted_avg_int(
                    [r.Proposition.Summarizes_Introduction for r in runs]
                ),
                Overall=weighted_avg_int([r.Proposition.Overall for r in runs]),
                Feedback="",
            ),
            Main_Points=MainPointsScores(
                Clarity=weighted_avg_int([r.Main_Points.Clarity for r in runs]),
                Hortatory_Universal_Truths=weighted_avg_int(
                    [r.Main_Points.Hortatory_Universal_Truths for r in runs]
                ),
                Proportional_and_Coexistent=weighted_avg_int(
                    [r.Main_Points.Proportional_and_Coexistent for r in runs]
                ),
                Exposition_Quality=weighted_avg_int(
                    [r.Main_Points.Exposition_Quality for r in runs]
                ),
                Illustration_Quality=weighted_avg_int(
                    [r.Main_Points.Illustration_Quality for r in runs]
                ),
                Application_Quality=weighted_avg_int(
                    [r.Main_Points.Application_Quality for r in runs]
                ),
                Overall=weighted_avg_int([r.Main_Points.Overall for r in runs]),
                Feedback="",
            ),
            Exegetical_Support=ExegeticalSupportScores(
                Alignment_with_Text=weighted_avg_int(
                    [r.Exegetical_Support.Alignment_with_Text for r in runs]
                ),
                Handles_Difficulties=weighted_avg_int(
                    [r.Exegetical_Support.Handles_Difficulties for r in runs]
                ),
                Proof_Accuracy_and_Clarity=weighted_avg_int(
                    [r.Exegetical_Support.Proof_Accuracy_and_Clarity for r in runs]
                ),
                Context_and_Genre_Considered=weighted_avg_int(
                    [r.Exegetical_Support.Context_and_Genre_Considered for r in runs]
                ),
                Not_Belabored=weighted_avg_int(
                    [r.Exegetical_Support.Not_Belabored for r in runs]
                ),
                Aids_Rather_Than_Impresses=weighted_avg_int(
                    [r.Exegetical_Support.Aids_Rather_Than_Impresses for r in runs]
                ),
                Overall=weighted_avg_int([r.Exegetical_Support.Overall for r in runs]),
                Feedback="",
            ),
            Application=ApplicationScores(
                Clear_and_Practical=weighted_avg_int(
                    [r.Application.Clear_and_Practical for r in runs]
                ),
                Redemptive_Focus=weighted_avg_int(
                    [r.Application.Redemptive_Focus for r in runs]
                ),
                Mandate_vs_Idea_Distinction=weighted_avg_int(
                    [r.Application.Mandate_vs_Idea_Distinction for r in runs]
                ),
                Passage_Supported=weighted_avg_int(
                    [r.Application.Passage_Supported for r in runs]
                ),
                Overall=weighted_avg_int([r.Application.Overall for r in runs]),
                Feedback="",
            ),
            Illustrations=IllustrationsScores(
                Lived_Body_Detail=weighted_avg_int(
                    [r.Illustrations.Lived_Body_Detail for r in runs]
                ),
                Strengthens_Points=weighted_avg_int(
                    [r.Illustrations.Strengthens_Points for r in runs]
                ),
                Proportion=weighted_avg_int([r.Illustrations.Proportion for r in runs]),
                Overall=weighted_avg_int([r.Illustrations.Overall for r in runs]),
                Feedback="",
            ),
            Conclusion=ConclusionScores(
                Summary=weighted_avg_int([r.Conclusion.Summary for r in runs]),
                Compelling_Exhortation=weighted_avg_int(
                    [r.Conclusion.Compelling_Exhortation for r in runs]
                ),
                Climax=weighted_avg_int([r.Conclusion.Climax for r in runs]),
                Pointed_End=weighted_avg_int([r.Conclusion.Pointed_End for r in runs]),
                Overall=weighted_avg_int([r.Conclusion.Overall for r in runs]),
                Feedback="",
            ),
            Strengths=runs[0].Strengths,  # Placeholder, will be harmonized
            Growth_Areas=runs[0].Growth_Areas,
            Next_Steps=runs[0].Next_Steps,
            Scoring_Confidence=sum(
                c * w for c, w in zip(confidences, weights)
            ),  # Weighted average
        )

        # Prepare harmonization input with all runs' feedback + confidence scores
        harmonize_input = {
            "averaged_scores": averaged.model_dump(),
            "runs_feedback": [
                {
                    "run_id": i + 1,
                    "confidence": r.Scoring_Confidence,
                    "introduction_feedback": r.Introduction.Feedback,
                    "proposition_feedback": r.Proposition.Feedback,
                    "main_points_feedback": r.Main_Points.Feedback,
                    "exegetical_support_feedback": r.Exegetical_Support.Feedback,
                    "application_feedback": r.Application.Feedback,
                    "illustrations_feedback": r.Illustrations.Feedback,
                    "conclusion_feedback": r.Conclusion.Feedback,
                    "strengths": r.Strengths,
                    "growth_areas": r.Growth_Areas,
                    "next_steps": r.Next_Steps,
                }
                for i, r in enumerate(runs)
            ],
        }

        harmonize_prompt = (
            f"{self.prompts.HARMONIZE_INSTRUCTIONS}\\n\\n"
            f"Harmonization Input:\\n{json.dumps(harmonize_input, ensure_ascii=False, indent=2)}"
        )

        try:
            with self._upload_indicator(message="Harmonizing multi-run feedback"):
                harmonized_feedback_data = self.provider.generate_structured(
                    prompt=harmonize_prompt,
                    response_schema=SermonScoringStep2Raw,
                    system=self.prompts.HARMONIZE_SYSTEM_PROMPT,
                    model=self.model,
                )
            harmonized_feedback = SermonScoringStep2Raw(**harmonized_feedback_data)

            # Merge harmonized feedback into averaged scores
            averaged.Introduction.Feedback = (
                harmonized_feedback.Introduction.Feedback or ""
            )
            averaged.Proposition.Feedback = (
                harmonized_feedback.Proposition.Feedback or ""
            )
            averaged.Main_Points.Feedback = (
                harmonized_feedback.Main_Points.Feedback or ""
            )
            averaged.Exegetical_Support.Feedback = (
                harmonized_feedback.Exegetical_Support.Feedback or ""
            )
            averaged.Application.Feedback = (
                harmonized_feedback.Application.Feedback or ""
            )
            averaged.Illustrations.Feedback = (
                harmonized_feedback.Illustrations.Feedback or ""
            )
            averaged.Conclusion.Feedback = harmonized_feedback.Conclusion.Feedback or ""
            averaged.Strengths = harmonized_feedback.Strengths or []
            averaged.Growth_Areas = harmonized_feedback.Growth_Areas or []
            averaged.Next_Steps = harmonized_feedback.Next_Steps or []
        except Exception as exc:
            print(
                f"[sermons] Warning: harmonization failed, using first run's feedback: {exc}"
            )
            # Fallback: use first run's feedback
            averaged.Introduction.Feedback = runs[0].Introduction.Feedback or ""
            averaged.Proposition.Feedback = runs[0].Proposition.Feedback or ""
            averaged.Main_Points.Feedback = runs[0].Main_Points.Feedback or ""
            averaged.Exegetical_Support.Feedback = (
                runs[0].Exegetical_Support.Feedback or ""
            )
            averaged.Application.Feedback = runs[0].Application.Feedback or ""
            averaged.Illustrations.Feedback = runs[0].Illustrations.Feedback or ""
            averaged.Conclusion.Feedback = runs[0].Conclusion.Feedback or ""

        # Convert to full SermonScoringStep2 and apply post-processing
        scoring = SermonScoringStep2(
            Introduction=averaged.Introduction,
            Proposition=averaged.Proposition,
            Main_Points=averaged.Main_Points,
            Exegetical_Support=averaged.Exegetical_Support,
            Application=averaged.Application,
            Illustrations=averaged.Illustrations,
            Conclusion=averaged.Conclusion,
            Strengths=averaged.Strengths,
            Growth_Areas=averaged.Growth_Areas,
            Next_Steps=averaged.Next_Steps,
            Scoring_Confidence=averaged.Scoring_Confidence,
        )

        # Apply calibration, compute aggregates, apply duration penalty, generate aggregate feedback
        scoring = self._apply_strict_calibration(scoring, extraction)
        scoring.Aggregated_Summary = self._compute_aggregates(scoring, extraction)

        if extraction.audio_duration is not None:
            scoring.Aggregated_Summary = self._apply_duration_penalty(
                scoring.Aggregated_Summary, extraction.audio_duration
            )

        # Generate aggregate feedback (final step)
        step2_dump = scoring.model_dump(exclude={"Aggregated_Summary_Feedback"})
        extraction_json = json.dumps(extraction.model_dump(), ensure_ascii=False)

        duration_info = ""
        if extraction.audio_duration is not None:
            duration_minutes = extraction.audio_duration / 60.0
            duration_info = f"\\n\\nSermon Duration: {duration_minutes:.1f} minutes"
            if (
                scoring.Aggregated_Summary
                and scoring.Aggregated_Summary.duration_penalty
            ):
                duration_info += f" (penalty applied: {scoring.Aggregated_Summary.duration_penalty:.2f} points)"

        agg_prompt = (
            f"{self.prompts.AGG_SUMMARY_INSTRUCTIONS}\\n\\n"
            f"Multi-run note: This evaluation averaged {len(runs)} independent scoring runs with confidence-weighted harmonization.\\n\\n"
            f"Step 1 JSON:\\n{extraction_json}\\n\\n"
            f"Step 2 JSON:\\n{json.dumps(step2_dump, ensure_ascii=False)}\\n\\n"
            f"Aggregated Summary JSON:\\n"
            f"{json.dumps(scoring.Aggregated_Summary.model_dump() if scoring.Aggregated_Summary else {}, ensure_ascii=False)}"
            f"{duration_info}"
        )

        try:
            with self._upload_indicator(message="Generating aggregate feedback"):
                agg_feedback_data = self.provider.generate_structured(
                    prompt=agg_prompt,
                    response_schema=AggregatedSummaryFeedback,
                    system=self.prompts.AGG_SUMMARY_SYSTEM_PROMPT,
                    model=self.model,
                )
            scoring.Aggregated_Summary_Feedback = AggregatedSummaryFeedback(
                **agg_feedback_data
            )
        except Exception as exc:
            print(f"[sermons] Warning: could not generate aggregate feedback: {exc}")

        return scoring

    # ----------- Aggregates computation -----------
    @staticmethod
    def _clamp(v: float, lo: float = 1.0, hi: float = 5.0) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _avg(vals):
        lst = [v for v in vals if v is not None]
        return sum(lst) / len(lst) if lst else 1.0

    def _compute_aggregates(
        self, scoring: SermonScoringStep2, extraction: SermonExtractionStep1
    ) -> AggregatedSummary:
        # Component rollups (all on 1–5)
        textual_fidelity = self._avg(
            [
                scoring.Exegetical_Support.Alignment_with_Text,
                scoring.Exegetical_Support.Handles_Difficulties,
                scoring.Exegetical_Support.Proof_Accuracy_and_Clarity,
                scoring.Exegetical_Support.Context_and_Genre_Considered,
            ]
        )

        proposition_clarity = self._avg(
            [
                scoring.Proposition.Principle_and_Application_Wed,
                scoring.Proposition.Establishes_Main_Theme,
                scoring.Proposition.Summarizes_Introduction,
            ]
        )

        introduction = self._avg(
            [
                scoring.Introduction.FCF_Introduced,
                scoring.Introduction.Arouses_Attention,
            ]
        )

        application_effectiveness = self._avg(
            [
                scoring.Application.Clear_and_Practical,
                scoring.Application.Redemptive_Focus,
                scoring.Application.Mandate_vs_Idea_Distinction,
                scoring.Application.Passage_Supported,
                scoring.Main_Points.Application_Quality,
            ]
        )

        structure_cohesion = self._avg(
            [
                scoring.Main_Points.Proportional_and_Coexistent,
                scoring.Conclusion.Summary,
                scoring.Conclusion.Compelling_Exhortation,
                scoring.Conclusion.Climax,
                scoring.Conclusion.Pointed_End,
            ]
        )

        illustrations = self._avg(
            [
                scoring.Main_Points.Illustration_Quality,
                scoring.Illustrations.Lived_Body_Detail,
                scoring.Illustrations.Strengthens_Points,
                scoring.Illustrations.Proportion,
            ]
        )

        # Weighted average per "Pillars First" scheme:
        # Emphasizes textual fidelity, application, and structure (68%),
        # with proposition (12%) and intro/illustrations (10% each) supporting.
        # Weights align with Evaluation Pillars in the framework.
        weights = {
            "textual_fidelity": 0.24,
            "application_effectiveness": 0.24,
            "structure_cohesion": 0.20,
            "proposition_clarity": 0.12,
            "illustrations": 0.10,
            "introduction": 0.10,
        }
        overall_base = (
            textual_fidelity * weights["textual_fidelity"]
            + application_effectiveness * weights["application_effectiveness"]
            + structure_cohesion * weights["structure_cohesion"]
            + proposition_clarity * weights["proposition_clarity"]
            + illustrations * weights["illustrations"]
            + introduction * weights["introduction"]
        )
        overall = self._clamp(overall_base)

        # Round to two decimals per framework guidance
        def r2(x: float) -> float:
            return float(f"{x:.2f}")

        return AggregatedSummary(
            Textual_Fidelity=r2(textual_fidelity),
            Proposition_Clarity=r2(proposition_clarity),
            Introduction=r2(introduction),
            Application_Effectiveness=r2(application_effectiveness),
            Structure_Cohesion=r2(structure_cohesion),
            Illustrations=r2(illustrations),
            Overall_Impact_Base=r2(overall_base),
            Overall_Impact=r2(overall),
            duration_penalty=None,  # Will be populated by _apply_duration_penalty
        )

    def _apply_duration_penalty(
        self,
        aggregated: AggregatedSummary,
        duration_seconds: Optional[float],
    ) -> AggregatedSummary:
        """Apply penalty for sermons outside 35-50 minute optimal range.

        Penalty formula:
        - < 35 minutes: penalty = min((35 - minutes) / 10.0, 1.0)
        - > 50 minutes: penalty = min((minutes - 50) / 15.0, 1.0)

        Max penalty is 1.0 point deduction from Overall_Impact (clamped to 1.0 minimum).
        """
        if duration_seconds is None:
            return aggregated

        duration_minutes = duration_seconds / 60.0
        penalty = 0.0

        if duration_minutes < 35:
            penalty = min((35 - duration_minutes) / 10.0, 1.0)
        elif duration_minutes > 50:
            penalty = min((duration_minutes - 50) / 15.0, 1.0)

        if penalty > 0:
            aggregated.duration_penalty = round(penalty, 2)
            aggregated.Overall_Impact = max(1.0, aggregated.Overall_Impact - penalty)
            aggregated.Overall_Impact = round(aggregated.Overall_Impact, 2)

        return aggregated

    # ----------- Post-processing: stricter calibration -----------
    def _apply_strict_calibration(
        self, scoring: SermonScoringStep2, extraction: SermonExtractionStep1
    ) -> SermonScoringStep2:
        """Downshift inflated scores using evidence from Step 1.

        Heuristics (conservative, integer outputs 1–5):
        - No explicit proposition -> Proposition sub-scores max 2 by default.
          CONDITIONAL SOFTENING: If proposition missing BUT Step 1 shows strong evidence
          (specific FCF, ≥3 body points OR ≥2 with subpoints, ≥2/3 points with concrete
          applications, ≥2 hortatory points, cohesion indicators), cap is raised to 3
          and placeholder bias is skipped for Proposition category.
        - No explicit conclusion -> Conclusion sub-scores max 2.
        - FCF missing/too vague -> Introduction.FCF_Introduced max 2.
          Vagueness criteria: <20 chars OR <6 words OR unqualified generic singletons
          OR vague phrases. Specificity signals (qualifiers, concrete needs) override.
        - Body points: if >50% lack Applications -> Main_Points.Application_Quality = min(curr, 2).
        - Body points: if >50% lack Illustrations -> Main_Points.Illustration_Quality = min(curr, 2) and Illustrations.* subs max 3.
        - If Body has <2 points -> Main_Points.Proportional_and_Coexistent max 2 and Structure cohesion later reflects via aggregates.
        - If any Step 1 field carries canonical placeholder, bias related category Overall down by 1 (min 1).
          Exception: placeholder bias skipped for Proposition when conditional thresholds met.
        After changes, recompute each section's Overall as ceil(avg(subs)).
        """
        from math import ceil

        def clamp_int(v: int, lo: int = 1, hi: int = 5) -> int:
            return max(lo, min(hi, int(v)))

        # Convenience references
        prop_text = (extraction.Proposition or "").strip().lower()
        concl_text = (extraction.Conclusion or "").strip().lower()
        fcf_original = (extraction.Fallen_Condition_Focus.FCF or "").strip().lower()
        body = extraction.Body or []

        # Helper: check if FCF is specific (not vague)
        def is_fcf_specific(fcf_str: str) -> bool:
            """Layered FCF specificity check.
            
            Returns False if:
            - Length < 20 chars OR < 6 words
            - Contains unqualified generic singletons (exact match as standalone word)
            - Contains vague phrases
            Returns True if:
            - Contains specificity signals (qualifiers, concrete needs)
            """
            if not fcf_str or len(fcf_str) < 20:
                return False
            word_count = len(fcf_str.split())
            if word_count < 6:
                return False
                        
            # Generic singletons (exact word match)
            generic_singletons = [
                "sin", "sinfulness", "brokenness", "struggle", "temptation",
                "fear", "doubt", "pride", "guilt", "shame", "idolatry",
                "loneliness", "suffering", "pain"
            ]
            words = fcf_str.split()
            # Check if FCF is ONLY a singleton (unqualified)
            if word_count <= 2 and any(w in generic_singletons for w in words):
                return False
            
            # Vague phrases
            vague_phrases = [
                "we all struggle", "people struggle", "general", "human condition",
                "broken world", "sin nature", "common issue", "life is hard",
                "try harder"
            ]
            if any(vp in fcf_str for vp in vague_phrases):
                return False
            
            # Specificity signals (qualifiers and concrete needs)
            qualifiers = ["of ", "for ", "in ", "under ", "against ", "before ", "toward ", "from "]
            concrete_needs = [
                "fear of man", "works-righteousness", "self-reliant", "self reliant",
                "partiality", "bitterness", "envy", "sexual immorality", "resentment",
                "control", "anxiety", "legalism", "unbelief", "despair", "self-righteousness",
                "self righteousness", "performance", "approval", "abandonment"
            ]
            has_qualifier = any(q in fcf_str for q in qualifiers)
            has_concrete = any(cn in fcf_str for cn in concrete_needs)
            
            return has_qualifier or has_concrete
        
        # Helper: check if point text has hortatory cues
        def has_hortatory_cues(point_text: str, summary_text: str) -> bool:
            """Check for hortatory language (imperative/exhortation) vs mere recap."""
            combined = (point_text + " " + summary_text).lower()
            # Hortatory signals
            hortatory = [
                "should", "must", "ought", "need to", "called to", "repent", "obey",
                "trust", "believe", "follow", "pursue", "reject", "embrace",
                "because", "therefore", "so that", "thus", "hence"
            ]
            # Recap signals (negative)
            recap = ["verses", "talks about", "is about", "discusses", "mentions", "says that"]
            
            has_hort = any(h in combined for h in hortatory)
            has_recap = any(r in combined for r in recap)
            
            return has_hort and not has_recap
        
        # Helper: check if applications are concrete
        def has_concrete_application(app_list: list) -> bool:
            """Check if application list contains concrete action verbs/nouns."""
            if not app_list:
                return False
            app_text = " ".join(app_list).lower()
            concrete_verbs = [
                "repent", "confess", "trust", "believe", "pray", "reconcile",
                "forgive", "serve", "love", "honor", "submit", "obey", "seek",
                "rest", "rejoice", "give", "share", "pursue", "reject", "turn"
            ]
            return any(cv in app_text for cv in concrete_verbs)
        
        # 1) Proposition present? Check conditional thresholds for softening
        proposition_missing = "no explicit proposition" in prop_text or not prop_text
        conditional_softening_applies = False
        fcf_specific = is_fcf_specific(fcf_original)
        
        if proposition_missing:
            # Check evidence thresholds
            
            body_count_ok = len(body) >= 3 or (
                len(body) >= 2 and any(p.Subpoints for p in body)
            )
            
            points_with_concrete_apps = sum(
                1 for p in body if has_concrete_application(p.Application or [])
            )
            apps_present = len(body) > 0 and (points_with_concrete_apps / len(body)) >= 0.67
            
            hortatory_count = sum(
                1 for p in body
                if has_hortatory_cues(p.Point or "", p.Summary or "")
            )
            hortatory_ok = hortatory_count >= 2
            
            # Cohesion: check for lexical overlap and conclusion presence
            conclusion_present = "no explicit conclusion" not in concl_text and concl_text
            # Simple lexical overlap: at least 2 meaningful words (>3 chars) shared between FCF and points
            fcf_words = set(w for w in fcf_original.lower().split() if len(w) > 3)
            point_words = set(
                w for p in body for w in (p.Point or "").lower().split() if len(w) > 3
            )
            overlap_ok = len(fcf_words & point_words) >= 2
            cohesion_ok = conclusion_present and overlap_ok
            
            # Apply conditional softening if all thresholds met
            conditional_softening_applies = (
                fcf_specific and body_count_ok and apps_present and hortatory_ok and cohesion_ok
            )
            
            # Cap Proposition scores: ≤3 if conditional, ≤2 if strict
            cap = 3 if conditional_softening_applies else 2
            s = scoring.Proposition
            s.Principle_and_Application_Wed = clamp_int(
                min(s.Principle_and_Application_Wed, cap)
            )
            s.Establishes_Main_Theme = clamp_int(min(s.Establishes_Main_Theme, cap))
            s.Summarizes_Introduction = clamp_int(min(s.Summarizes_Introduction, cap))

        # 2) Conclusion present?
        if "no explicit conclusion" in concl_text or not concl_text:
            s = scoring.Conclusion
            s.Summary = clamp_int(min(s.Summary, 2))
            s.Compelling_Exhortation = clamp_int(min(s.Compelling_Exhortation, 2))
            s.Climax = clamp_int(min(s.Climax, 2))
            s.Pointed_End = clamp_int(min(s.Pointed_End, 2))

        # 3) FCF specificity (use improved layered check)
        if not fcf_specific:
            scoring.Introduction.FCF_Introduced = clamp_int(
                min(scoring.Introduction.FCF_Introduced, 2)
            )

        # 4) Body applications / illustrations
        total_points = len(body)
        empty_apps = sum(1 for p in body if not (p.Application or []))
        empty_illus = sum(1 for p in body if not (p.Illustrations or []))
        if total_points > 0:
            if empty_apps / total_points > 0.5:
                scoring.Main_Points.Application_Quality = clamp_int(
                    min(scoring.Main_Points.Application_Quality, 2)
                )
            if empty_illus / total_points > 0.5:
                scoring.Main_Points.Illustration_Quality = clamp_int(
                    min(scoring.Main_Points.Illustration_Quality, 2)
                )
                # Also constrain Illustrations category a bit if globally sparse
                scoring.Illustrations.Lived_Body_Detail = clamp_int(
                    min(scoring.Illustrations.Lived_Body_Detail, 3)
                )
                scoring.Illustrations.Strengthens_Points = clamp_int(
                    min(scoring.Illustrations.Strengthens_Points, 3)
                )
                scoring.Illustrations.Proportion = clamp_int(
                    min(scoring.Illustrations.Proportion, 3)
                )

        # 5) Too few points harms proportional/coexistent
        if total_points < 2:
            scoring.Main_Points.Proportional_and_Coexistent = clamp_int(
                min(scoring.Main_Points.Proportional_and_Coexistent, 2)
            )

        # 6) If placeholders present, downshift related Overall by 1 later
        #    Exception: skip Proposition bias when conditional softening applies
        placeholder_bias = 0
        proposition_bias_applies = (
            ("no explicit proposition" in prop_text or not prop_text)
            and not conditional_softening_applies
        )
        placeholder_bias += 1 if proposition_bias_applies else 0
        placeholder_bias += (
            1 if ("no explicit conclusion" in concl_text or not concl_text) else 0
        )

        # Recompute per-section Overall as ceil(avg of sub-scores) then apply placeholder bias
        def recompute_overall(values: list[int], current_overall: int) -> int:
            avg_val = sum(values) / max(1, len(values))
            recomputed = clamp_int(ceil(avg_val))
            if placeholder_bias and current_overall >= recomputed:
                recomputed = clamp_int(recomputed - 1)
            return recomputed

        scoring.Introduction.Overall = recompute_overall(
            [
                scoring.Introduction.FCF_Introduced,
                scoring.Introduction.Arouses_Attention,
            ],
            scoring.Introduction.Overall,
        )
        scoring.Proposition.Overall = recompute_overall(
            [
                scoring.Proposition.Principle_and_Application_Wed,
                scoring.Proposition.Establishes_Main_Theme,
                scoring.Proposition.Summarizes_Introduction,
            ],
            scoring.Proposition.Overall,
        )
        scoring.Main_Points.Overall = recompute_overall(
            [
                scoring.Main_Points.Clarity,
                scoring.Main_Points.Hortatory_Universal_Truths,
                scoring.Main_Points.Proportional_and_Coexistent,
                scoring.Main_Points.Exposition_Quality,
                scoring.Main_Points.Illustration_Quality,
                scoring.Main_Points.Application_Quality,
            ],
            scoring.Main_Points.Overall,
        )
        scoring.Exegetical_Support.Overall = recompute_overall(
            [
                scoring.Exegetical_Support.Alignment_with_Text,
                scoring.Exegetical_Support.Handles_Difficulties,
                scoring.Exegetical_Support.Proof_Accuracy_and_Clarity,
                scoring.Exegetical_Support.Context_and_Genre_Considered,
                scoring.Exegetical_Support.Not_Belabored,
                scoring.Exegetical_Support.Aids_Rather_Than_Impresses,
            ],
            scoring.Exegetical_Support.Overall,
        )
        scoring.Application.Overall = recompute_overall(
            [
                scoring.Application.Clear_and_Practical,
                scoring.Application.Redemptive_Focus,
                scoring.Application.Mandate_vs_Idea_Distinction,
                scoring.Application.Passage_Supported,
            ],
            scoring.Application.Overall,
        )
        scoring.Illustrations.Overall = recompute_overall(
            [
                scoring.Illustrations.Lived_Body_Detail,
                scoring.Illustrations.Strengthens_Points,
                scoring.Illustrations.Proportion,
            ],
            scoring.Illustrations.Overall,
        )
        scoring.Conclusion.Overall = recompute_overall(
            [
                scoring.Conclusion.Summary,
                scoring.Conclusion.Compelling_Exhortation,
                scoring.Conclusion.Climax,
                scoring.Conclusion.Pointed_End,
            ],
            scoring.Conclusion.Overall,
        )

        return scoring

    # ----------- Audio support (Gemini only) -----------
    def _load_audio_cache(self) -> Dict[str, Any]:
        try:
            return json.loads(self.audio_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_audio_cache(self, cache: Dict[str, Any]) -> None:
        self.audio_cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
            print(
                f"[sermons] [cache] Using previously uploaded file:\n  {abs_path} -> {remote_id}"
            )
            # Try fetching a live file object so contents include a valid file reference
            file_obj = None
            try:
                if hasattr(self.provider, "get_file"):
                    file_obj = self.provider.get_file(remote_id)  # type: ignore[attr-defined]
            except Exception as e:
                print(
                    f"[sermons] [cache] Could not fetch file object for {remote_id}: {e}"
                )
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
        print(
            f"[sermons] [upload] Uploading to Gemini Files API:\n  {abs_path} ({size_str})"
        )
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

        # Extract duration before upload
        duration = self.get_audio_duration(audio_path)
        if duration:
            print(f"[sermons] Audio duration: {duration / 60.0:.1f} minutes")

        file_id, file_obj = self.upload_or_get_gemini_file(audio_path)
        # Some cached entries may only return the id; try to fetch the object now
        if file_obj is None and hasattr(self.provider, "get_file"):
            try:
                file_obj = self.provider.get_file(file_id)  # type: ignore[attr-defined]
            except Exception as e:
                print(
                    f"[sermons] Warning: could not retrieve file object for {file_id}; proceeding with id only. {e}"
                )
        # Construct Gemini content: instruction + file
        instruction = self.prompts.EXTRACTION_INSTRUCTIONS_AUDIO
        system = self.prompts.EXTRACTION_SYSTEM_PROMPT
        # Use two-part content: instruction and file reference
        with self._upload_indicator(message="Step 1: Extracting structure from audio"):
            data = self.provider.generate_structured_with_contents(
                contents=[instruction, file_obj or file_id],
                response_schema=SermonExtractionStep1,
                system=system,
                model=self.model,
            )
        extraction = SermonExtractionStep1(**data)
        extraction.audio_duration = duration  # Populate duration field
        return extraction

    # ----------- Helpers -----------
    @staticmethod
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        text = text.strip()
        # Some providers may wrap JSON in code fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]  # remove 'json' label
        try:
            return json.loads(text)
        except Exception:
            # Last resort: attempt to locate first and last braces
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                return json.loads(text[first : last + 1])
            raise


__all__ = [
    "SermonEvaluationEngine",
]
