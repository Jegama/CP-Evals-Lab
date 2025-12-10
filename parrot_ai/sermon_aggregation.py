"""Aggregate score computation for sermon evaluation.

Computes weighted aggregate metrics and applies duration penalties.
"""

from __future__ import annotations

from typing import Optional

from .evaluation_schemas import (
    SermonExtractionStep1,
    SermonScoringStep2,
    AggregatedSummary,
)


class SermonAggregator:
    """Computes aggregate scores and applies duration penalties."""

    @staticmethod
    def clamp(v: float, lo: float = 1.0, hi: float = 5.0) -> float:
        """Clamp float value to [lo, hi] range."""
        return max(lo, min(hi, v))

    @staticmethod
    def avg(vals):
        """Calculate average of non-None values."""
        lst = [v for v in vals if v is not None]
        return sum(lst) / len(lst) if lst else 1.0

    def compute_aggregates(
        self, scoring: SermonScoringStep2, extraction: SermonExtractionStep1
    ) -> AggregatedSummary:
        """Compute aggregated summary scores with weighted average.

        Uses "Pillars First" weighting scheme:
        - Textual fidelity: 24%
        - Application effectiveness: 24%
        - Structure cohesion: 20%
        - Proposition clarity: 12%
        - Illustrations: 10%
        - Introduction: 10%
        """
        # Component rollups (all on 1–5 scale)
        textual_fidelity = self.avg(
            [
                scoring.Exegetical_Support.Alignment_with_Text,
                scoring.Exegetical_Support.Handles_Difficulties,
                scoring.Exegetical_Support.Proof_Accuracy_and_Clarity,
                scoring.Exegetical_Support.Context_and_Genre_Considered,
            ]
        )

        proposition_clarity = self.avg(
            [
                scoring.Proposition.Principle_and_Application_Wed,
                scoring.Proposition.Establishes_Main_Theme,
                scoring.Proposition.Summarizes_Introduction,
            ]
        )

        introduction = self.avg(
            [
                scoring.Introduction.FCF_Introduced,
                scoring.Introduction.Arouses_Attention,
            ]
        )

        application_effectiveness = self.avg(
            [
                scoring.Application.Clear_and_Practical,
                scoring.Application.Redemptive_Focus,
                scoring.Application.Mandate_vs_Idea_Distinction,
                scoring.Application.Passage_Supported,
                scoring.Main_Points.Application_Quality,
            ]
        )

        structure_cohesion = self.avg(
            [
                scoring.Main_Points.Proportional_and_Coexistent,
                scoring.Conclusion.Summary,
                scoring.Conclusion.Compelling_Exhortation,
                scoring.Conclusion.Climax,
                scoring.Conclusion.Pointed_End,
            ]
        )

        illustrations = self.avg(
            [
                scoring.Main_Points.Illustration_Quality,
                scoring.Illustrations.Lived_Body_Detail,
                scoring.Illustrations.Strengthens_Points,
                scoring.Illustrations.Proportion,
            ]
        )

        # Weighted average
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
        overall = self.clamp(overall_base)

        # Round to two decimals
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
            duration_penalty=None,
        )

    def apply_duration_penalty(
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


__all__ = ["SermonAggregator"]
