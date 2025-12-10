"""Multi-run harmonization logic for sermon evaluation.

Implements self-consistency with confidence-weighted averaging and LLM-based
feedback synthesis.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evaluation_schemas import (
    SermonExtractionStep1,
    SermonScoringStep2,
    SermonScoringStep2Raw,
    AggregatedSummaryFeedback,
)
from .sermon_aggregation import SermonAggregator
from .sermon_calibration import SermonScoreCalibrator
from .sermon_audio_utils import AudioFileManager


class SermonHarmonizer:
    """Handles multi-run scoring and harmonization."""

    def __init__(self, provider: Any, model: str, prompts: Any):
        self.provider = provider
        self.model = model
        self.prompts = prompts
        self.aggregator = SermonAggregator()
        self.calibrator = SermonScoreCalibrator()
        self.audio_manager = AudioFileManager()

    def score_single_run(
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

    def score_multi_run(
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

        seeds = [1689, 2024, 3141, 4567, 5000, 6789, 7890, 8888, 9999]
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

            with ThreadPoolExecutor(max_workers=min(needed, 5)) as executor:
                futures = {
                    executor.submit(
                        self.score_single_run, extraction, audio_file_obj, s
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

        return self.harmonize_runs(runs, extraction, audio_file_obj)

    def harmonize_runs(
        self,
        runs: List[SermonScoringStep2Raw],
        extraction: SermonExtractionStep1,
        audio_file_obj: Optional[Any],
    ) -> SermonScoringStep2:
        """Average numeric scores with confidence weighting and harmonize feedback via LLM.

        Implements self-consistency with confidence-based weighting:
        - Higher Scoring_Confidence runs contribute more to averaged scores
        - Harmonization LLM synthesizes feedback from all runs
        """
        print(f"[sermons] Harmonizing {len(runs)} scoring runs...")

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
        from .evaluation_schemas import (
            IntroductionScores,
            PropositionScores,
            MainPointsScores,
            ExegeticalSupportScores,
            ApplicationScores,
            IllustrationsScores,
            ConclusionScores,
        )

        averaged = SermonScoringStep2Raw(
            Introduction=IntroductionScores(
                FCF_Introduced=weighted_avg_int(
                    [r.Introduction.FCF_Introduced for r in runs]
                ),
                Arouses_Attention=weighted_avg_int(
                    [r.Introduction.Arouses_Attention for r in runs]
                ),
                Overall=weighted_avg_int([r.Introduction.Overall for r in runs]),
                Feedback="",
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
            Strengths=runs[0].Strengths,
            Growth_Areas=runs[0].Growth_Areas,
            Next_Steps=runs[0].Next_Steps,
            Scoring_Confidence=sum(c * w for c, w in zip(confidences, weights)),
        )

        # Harmonize feedback via LLM
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
            with self.audio_manager.upload_indicator(
                message="Harmonizing multi-run feedback"
            ):
                harmonized_feedback_data = self.provider.generate_structured(
                    prompt=harmonize_prompt,
                    response_schema=SermonScoringStep2Raw,
                    system=self.prompts.HARMONIZE_SYSTEM_PROMPT,
                    model=self.model,
                )
            harmonized_feedback = SermonScoringStep2Raw(**harmonized_feedback_data)

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
            averaged.Introduction.Feedback = runs[0].Introduction.Feedback or ""
            averaged.Proposition.Feedback = runs[0].Proposition.Feedback or ""
            averaged.Main_Points.Feedback = runs[0].Main_Points.Feedback or ""
            averaged.Exegetical_Support.Feedback = (
                runs[0].Exegetical_Support.Feedback or ""
            )
            averaged.Application.Feedback = runs[0].Application.Feedback or ""
            averaged.Illustrations.Feedback = runs[0].Illustrations.Feedback or ""
            averaged.Conclusion.Feedback = runs[0].Conclusion.Feedback or ""

        # Convert to full SermonScoringStep2
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

        # Apply calibration, aggregates, duration penalty
        scoring = self.calibrator.apply_strict_calibration(scoring, extraction)
        scoring.Aggregated_Summary = self.aggregator.compute_aggregates(
            scoring, extraction
        )

        if extraction.audio_duration is not None:
            scoring.Aggregated_Summary = self.aggregator.apply_duration_penalty(
                scoring.Aggregated_Summary, extraction.audio_duration
            )

        # Generate aggregate feedback
        self._generate_aggregate_feedback(scoring, extraction, len(runs))

        return scoring

    def _generate_aggregate_feedback(
        self,
        scoring: SermonScoringStep2,
        extraction: SermonExtractionStep1,
        num_runs: int,
    ) -> None:
        """Generate and attach aggregate feedback to scoring."""
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

        multi_run_note = (
            f"\\n\\nMulti-run note: This evaluation averaged {num_runs} independent scoring runs with confidence-weighted harmonization."
            if num_runs > 1
            else ""
        )

        agg_prompt = (
            f"{self.prompts.AGG_SUMMARY_INSTRUCTIONS}{multi_run_note}\\n\\n"
            f"Step 1 JSON:\\n{extraction_json}\\n\\n"
            f"Step 2 JSON:\\n{json.dumps(step2_dump, ensure_ascii=False)}\\n\\n"
            f"Aggregated Summary JSON:\\n"
            f"{json.dumps(scoring.Aggregated_Summary.model_dump() if scoring.Aggregated_Summary else {}, ensure_ascii=False)}"
            f"{duration_info}"
        )

        try:
            with self.audio_manager.upload_indicator(
                message="Generating aggregate feedback"
            ):
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


__all__ = ["SermonHarmonizer"]
