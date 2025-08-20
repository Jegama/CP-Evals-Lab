"""Pydantic schema models for evaluation results.

Separated from ``llm_evaluation`` during refactor so they can be imported
without pulling in OpenAI / provider logic.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class AdherenceModel(BaseModel):
    Core: int
    Secondary: int
    Tertiary_Handling: int
    Biblical_Basis: int
    Consistency: int
    Overall: int

class KindnessGentlenessModel(BaseModel):
    Core_Clarity_with_Kindness: int
    Pastoral_Sensitivity: int
    Secondary_Fairness: int
    Tertiary_Neutrality: int
    Tone: int
    Overall: int

class InterfaithSensitivityModel(BaseModel):
    Respect_and_Handling_Objections: int
    Objection_Acknowledgement: int
    Evangelism: int
    Gospel_Boldness: int
    Overall: int

class ArabicAccuracyDetailed(BaseModel):  # Arabic only
    Grammar_and_Syntax: int
    Theological_Nuance: int
    Contextual_Clarity: int
    Consistency_of_Terms: int
    Arabic_Purity: int
    Penalty_Reason: Optional[str] = None
    Overall: int

class EvaluationResultArabic(BaseModel):
    Adherence: AdherenceModel
    Kindness_and_Gentleness: KindnessGentlenessModel
    Interfaith_Sensitivity: InterfaithSensitivityModel
    Arabic_Accuracy: ArabicAccuracyDetailed

class EvaluationResultEnglish(BaseModel):
    Adherence: AdherenceModel
    Kindness_and_Gentleness: KindnessGentlenessModel
    Interfaith_Sensitivity: InterfaithSensitivityModel

__all__ = [
    'AdherenceModel',
    'KindnessGentlenessModel',
    'InterfaithSensitivityModel',
    'ArabicAccuracyDetailed',
    'EvaluationResultArabic',
    'EvaluationResultEnglish',
]
