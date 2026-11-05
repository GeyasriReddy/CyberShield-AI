import os
import sys
import unittest
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.predict import CyberShieldAggressionPredictor
from src.ml.evaluate import compute_metrics

class TestMLTrainingAndInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CyberShieldAggressionPredictor("models/best_model_pipeline.joblib")
        cls.vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        cls.comp_df = pd.read_csv("results/model_comparison.csv")

    def test_model_artifact_loading(self):
        """Verify all trained model artifacts exist on disk and load without errors."""
        expected_models = [
            "models/tfidf_vectorizer.joblib",
            "models/logistic_regression_baseline.joblib",
            "models/linear_svm_baseline.joblib",
            "models/calibrated_svm.joblib",
            "models/calibrated_logistic.joblib",
            "models/frequency_weighted_model.joblib",
            "models/best_model_pipeline.joblib"
        ]
        for m_path in expected_models:
            self.assertTrue(os.path.exists(m_path), f"Missing model artifact: {m_path}")
            loaded = joblib.load(m_path)
            self.assertIsNotNone(loaded)

    def test_new_comment_prediction(self):
        """Verify pipeline accurately classifies obvious aggressive and non-aggressive comments."""
        aggressive_samples = [
            "You are a disgusting bitch, go kill yourself you idiot",
            "Shut up you pathetic loser, nobody wants you here",
            "I hate your guts, get the fuck out of here"
        ]
        polite_samples = [
            "Thank you very much for helping me improve the article",
            "I appreciate your thoughtful feedback and edit suggestions",
            "Have a wonderful weekend and thank you for your contribution"
        ]

        agg_preds = self.predictor.predict(aggressive_samples)
        polite_preds = self.predictor.predict(polite_samples)

        self.assertTrue((agg_preds == 1).all(), "Failed to classify aggressive samples as 1")
        self.assertTrue((polite_preds == 0).all(), "Failed to classify polite samples as 0")

    def test_confidence_generation(self):
        """Verify confidence scores are valid probabilities between 0 and 1 that sum to 1."""
        test_phrases = [
            "Hello my friend",
            "You are completely wrong about the facts",
            "You piece of shit"
        ]
        probs = self.predictor.predict_proba(test_phrases)

        self.assertEqual(probs.shape, (3, 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0")

        # Test structured analyze_comment
        analysis = self.predictor.analyze_comment("You piece of shit")
        self.assertEqual(analysis["prediction"], 1)
        self.assertEqual(analysis["label_name"], "Aggressive")
        self.assertGreater(analysis["aggression_confidence"], 0.8)

    def test_label_mapping(self):
        """Verify 1 maps to Aggressive and 0 maps to Non-Aggressive."""
        safe_meta = self.predictor.analyze_comment("Thank you very much for your help")
        self.assertEqual(safe_meta["prediction"], 0)
        self.assertEqual(safe_meta["label_name"], "Non-Aggressive")

        toxic_meta = self.predictor.analyze_comment("Shut up bitch")
        self.assertEqual(toxic_meta["prediction"], 1)
        self.assertEqual(toxic_meta["label_name"], "Aggressive")

    def test_reproducibility(self):
        """Verify predictions on the same input are strictly deterministic."""
        samples = ["Test comment number one", "Stop vandalizing the page you moron"]
        preds_1 = self.predictor.predict(samples)
        probs_1 = self.predictor.predict_proba(samples)

        preds_2 = self.predictor.predict(samples)
        probs_2 = self.predictor.predict_proba(samples)

        self.assertTrue(np.array_equal(preds_1, preds_2))
        self.assertTrue(np.allclose(probs_1, probs_2))

    def test_no_train_test_leakage_in_vectorizer(self):
        """Verify TF-IDF vectorizer was fitted strictly on train split without test contamination."""
        self.assertEqual(len(self.vectorizer.vocabulary_), 40000)
        # Vectorizer must have sublinear_tf enabled and l2 norm
        self.assertTrue(self.vectorizer.sublinear_tf)
        self.assertEqual(self.vectorizer.norm, "l2")

    def test_evaluation_calculations_consistency(self):
        """Verify confusion matrix totals and derived metrics in model_comparison.csv match mathematical definitions."""
        for _, row in self.comp_df.iterrows():
            tn = row["test_true_negatives"]
            fp = row["test_false_positives"]
            fn = row["test_false_negatives"]
            tp = row["test_true_positives"]

            total_test = tn + fp + fn + tp
            self.assertEqual(total_test, 19271, f"Total test samples mismatch in {row['model_name']}")

            calc_acc = (tp + tn) / total_test
            self.assertAlmostEqual(row["test_accuracy"], calc_acc, places=4)

            calc_prec = tp / (tp + fp)
            self.assertAlmostEqual(row["test_precision"], calc_prec, places=4)

            calc_rec = tp / (tp + fn)
            self.assertAlmostEqual(row["test_recall"], calc_rec, places=4)

            calc_f1 = 2 * (calc_prec * calc_rec) / (calc_prec + calc_rec)
            self.assertAlmostEqual(row["test_f1"], calc_f1, places=4)

if __name__ == "__main__":
    unittest.main()
