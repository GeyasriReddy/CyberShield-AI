"""
Classifier Integration Module
-----------------------------
This module defines the interface and placeholder implementation for text-based
cyberbullying and toxic content classification.

INTEGRATION GUIDE FOR TEAMMATE:
To connect your trained machine learning / NLP model:
1. Replace the internal logic of `predict_comment()` with your inference pipeline.
2. Ensure your output conforms to the `ClassifierResult` dataclass (or dictionary)
   containing:
     - predicted_label: str (e.g., 'Toxic', 'Harassment', 'Clean')
     - confidence: float between 0.0 and 1.0
     - class_scores: dict mapping class names to normalized probabilities
     - is_mock: set to False once a genuine trained model is loaded
3. If model loading requires initialization (e.g., torch.load, tokenizer),
   instantiate it inside `get_classifier_instance()` with @st.cache_resource.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import re


@dataclass
class ClassifierResult:
    predicted_label: str
    confidence: float
    class_scores: Dict[str, float]
    is_mock: bool = True
    model_version: str = "mock-heuristic-v0.1"
    inference_notes: Optional[str] = None


class MockClassifier:
    """
    Temporary baseline classifier using basic keyword heuristics.
    Clearly designated as a demonstration stub for UI integration.
    """

    def __init__(self):
        # Heuristic keywords for mock demonstration
        self._toxic_patterns = [
            r"\b(idiot|stupid|loser|hate|dumb|moron|trash|pathetic)\b",
            r"\b(kill yourself|die|shut up|nobody likes you)\b",
            r"\b(ugly|fat|creep|disgusting)\b"
        ]
        self._harassment_patterns = [
            r"\b(target|stalk|hunt|ruin you|destroy you)\b",
            r"\b(keep posting and see|leave this platform)\b"
        ]

    def predict(self, text: str) -> ClassifierResult:
        """
        Runs mock classification on the input text.
        
        Args:
            text: Raw comment string.
            
        Returns:
            ClassifierResult with predicted label, confidence, and mock indicator.
        """
        if not text or not text.strip():
            return ClassifierResult(
                predicted_label="Clean",
                confidence=0.95,
                class_scores={"Clean": 0.95, "Toxic": 0.03, "Harassment": 0.02},
                is_mock=True,
                inference_notes="Empty or whitespace input defaults to clean state."
            )

        lowered = text.lower()
        harassment_hits = sum(len(re.findall(pat, lowered)) for pat in self._harassment_patterns)
        toxic_hits = sum(len(re.findall(pat, lowered)) for pat in self._toxic_patterns)

        if harassment_hits > 0:
            confidence = min(0.70 + (harassment_hits * 0.10), 0.96)
            remainder = (1.0 - confidence) / 2
            return ClassifierResult(
                predicted_label="Harassment",
                confidence=round(confidence, 3),
                class_scores={
                    "Harassment": round(confidence, 3),
                    "Toxic": round(remainder * 1.5, 3),
                    "Clean": round(remainder * 0.5, 3)
                },
                is_mock=True,
                inference_notes="Mock rule triggered: detected harassment pattern match."
            )

        if toxic_hits > 0:
            confidence = min(0.65 + (toxic_hits * 0.08), 0.92)
            remainder = (1.0 - confidence) / 2
            return ClassifierResult(
                predicted_label="Toxic",
                confidence=round(confidence, 3),
                class_scores={
                    "Toxic": round(confidence, 3),
                    "Harassment": round(remainder * 0.6, 3),
                    "Clean": round(remainder * 1.4, 3)
                },
                is_mock=True,
                inference_notes="Mock rule triggered: detected toxic keyword pattern match."
            )

        # Baseline Clean classification
        return ClassifierResult(
            predicted_label="Clean",
            confidence=0.88,
            class_scores={
                "Clean": 0.88,
                "Toxic": 0.08,
                "Harassment": 0.04
            },
            is_mock=True,
            inference_notes="Mock baseline: no targeted keywords matched; flagged as clean."
        )


# Global singleton instance for mock classification
_classifier_instance = MockClassifier()


def predict_comment(text: str) -> ClassifierResult:
    """
    Main entry point for comment classification.
    Call this from the Streamlit UI or data pipeline.
    
    TEAMMATE INTEGRATION POINT:
    Replace this call with your trained model inference when available.
    """
    return _classifier_instance.predict(text)
