"""
Classifier Integration Module
-----------------------------
Connects the trained machine learning pipeline (Calibrated Logistic Regression)
for text-based cyberbullying and aggression classification.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os
import re

# Model artifact path
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "best_model_pipeline.joblib"
)


@dataclass
class ClassifierResult:
    predicted_label: str
    confidence: float
    class_scores: Dict[str, float]
    is_mock: bool = False
    model_version: str = "Calibrated-Logistic-v1.0"
    inference_notes: Optional[str] = None


class ProductionMLClassifier:
    """
    Production classifier powered by teammate's trained Calibrated Logistic Regression pipeline.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        from src.ml.predict import CyberShieldAggressionPredictor
        self.predictor = CyberShieldAggressionPredictor(pipeline_path=model_path)
        self.version = "Calibrated-Logistic-v1.0 (88.7% Test Accuracy)"

    def predict(self, text: str) -> ClassifierResult:
        if not text or not text.strip():
            return ClassifierResult(
                predicted_label="Clean",
                confidence=0.99,
                class_scores={"Clean": 0.99, "Toxic": 0.01},
                is_mock=False,
                model_version=self.version,
                inference_notes="Empty or whitespace input defaults to clean baseline."
            )

        res = self.predictor.analyze_comment(text)
        agg_prob = res["aggression_confidence"]
        non_agg_prob = res["non_aggression_confidence"]
        is_aggressive = res["prediction"] == 1

        label = "Toxic" if is_aggressive else "Clean"
        conf = agg_prob if is_aggressive else non_agg_prob

        return ClassifierResult(
            predicted_label=label,
            confidence=round(conf, 4),
            class_scores={"Toxic": round(agg_prob, 4), "Clean": round(non_agg_prob, 4)},
            is_mock=False,
            model_version=self.version,
            inference_notes=f"Trained ML Model Inference: {res['severity_assessment']} (Calibrated probability: {conf:.2%})"
        )


class MockClassifier:
    """
    Fallback baseline classifier in case model weights are not loaded.
    """

    def predict(self, text: str) -> ClassifierResult:
        lowered = text.lower() if text else ""
        if any(w in lowered for w in ["idiot", "loser", "destroy", "kill", "hate"]):
            return ClassifierResult(
                predicted_label="Toxic",
                confidence=0.88,
                class_scores={"Toxic": 0.88, "Clean": 0.12},
                is_mock=True,
                model_version="mock-heuristic-v0.1",
                inference_notes="Fallback mock rule triggered."
            )
        return ClassifierResult(
            predicted_label="Clean",
            confidence=0.92,
            class_scores={"Clean": 0.92, "Toxic": 0.08},
            is_mock=True,
            model_version="mock-heuristic-v0.1",
            inference_notes="Fallback mock baseline clean."
        )


# Initialize singleton classifier (Production first, fallback to mock)
_classifier_instance = None

try:
    if os.path.exists(MODEL_PATH):
        _classifier_instance = ProductionMLClassifier(MODEL_PATH)
    else:
        _classifier_instance = MockClassifier()
except Exception as e:
    _classifier_instance = MockClassifier()


def predict_comment(text: str) -> ClassifierResult:
    """
    Main entry point for comment classification called by Streamlit UI and pipelines.
    """
    return _classifier_instance.predict(text)
