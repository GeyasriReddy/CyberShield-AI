import os
import sys
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_data_prep import cybershield_url_normalizer

class CyberShieldAggressionPredictor:
    """
    Inference pipeline for CyberShield AI text aggression detection.
    Predicts:
      - 1: Aggressive language
      - 0: Non-aggressive language
    Includes confidence calibration scores.
    """
    def __init__(self, pipeline_path="models/best_model_pipeline.joblib"):
        self.pipeline_path = pipeline_path
        self.pipeline = joblib.load(pipeline_path)

    def predict(self, texts):
        """
        Predicts binary aggression label (0 or 1).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        """
        Predicts calibrated probability distribution [P(non-agg), P(agg)].
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict_proba(texts)

    def analyze_comment(self, text):
        """
        Analyzes a single text comment and returns structured metadata.
        """
        probs = self.predict_proba([text])[0]
        agg_prob = float(probs[1])
        label = int(np.argmax(probs))

        if agg_prob >= 0.85:
            severity = "Severe Aggression"
        elif agg_prob >= 0.65:
            severity = "Moderate Aggression"
        elif agg_prob >= 0.45:
            severity = "Ambiguous / Low Risk"
        else:
            severity = "Non-Aggressive / Safe"

        return {
            "text": text,
            "prediction": label,
            "label_name": "Aggressive" if label == 1 else "Non-Aggressive",
            "aggression_confidence": round(agg_prob, 4),
            "non_aggression_confidence": round(float(probs[0]), 4),
            "severity_assessment": severity
        }

