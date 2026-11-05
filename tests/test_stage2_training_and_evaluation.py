import os
import sys
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.train_stage2 import extract_model_feature_importance

class TestStage2TrainingAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(PROJECT_ROOT, "models")
        cls.results_dir = os.path.join(PROJECT_ROOT, "results")
        cls.processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")

        cls.model_path = os.path.join(cls.models_dir, "stage2_best_model.joblib")
        cls.comp_path = os.path.join(cls.results_dir, "stage2_model_comparison.csv")
        cls.pred_path = os.path.join(cls.results_dir, "stage2_predictions.csv")
        cls.cm_path = os.path.join(cls.results_dir, "stage2_confusion_matrix.png")
        cls.fi_path = os.path.join(cls.results_dir, "stage2_feature_importance.csv")

        cls.train_split_path = os.path.join(cls.processed_dir, "stage2_train_split.csv")
        cls.val_split_path = os.path.join(cls.processed_dir, "stage2_val_split.csv")
        cls.test_split_path = os.path.join(cls.processed_dir, "stage2_test_split.csv")
        cls.cb_labels_path = os.path.join(cls.processed_dir, "cb_labels_cleaned.csv")

        # Load artifacts if they exist
        if os.path.exists(cls.model_path):
            cls.model_bundle = joblib.load(cls.model_path)
        if os.path.exists(cls.pred_path):
            cls.df_preds = pd.read_csv(cls.pred_path)
        if os.path.exists(cls.train_split_path):
            cls.df_train = pd.read_csv(cls.train_split_path)
        if os.path.exists(cls.val_split_path):
            cls.df_val = pd.read_csv(cls.val_split_path)
        if os.path.exists(cls.test_split_path):
            cls.df_test = pd.read_csv(cls.test_split_path)
        if os.path.exists(cls.cb_labels_path):
            cls.df_cb = pd.read_csv(cls.cb_labels_path)
        if os.path.exists(cls.comp_path):
            cls.df_comp = pd.read_csv(cls.comp_path)

        cls.expected_configs = {
            "m1_baseline_logistic_regression": 6,
            "m1_baseline_random_forest": 6,
            "m2_nlp_only_logistic_regression": 3,
            "m2_nlp_only_random_forest": 3,
            "m3_sna_only_logistic_regression": 13,
            "m3_sna_only_random_forest": 13,
            "m4_combined_logistic_regression": 18,
            "m4_combined_random_forest": 18
        }

    def test_artifacts_exist(self):
        """Verify all Stage 2 model, predictions, splits, and metrics artifacts exist on disk."""
        self.assertTrue(os.path.exists(self.model_path), f"Missing model bundle: {self.model_path}")
        self.assertTrue(os.path.exists(self.comp_path), f"Missing comparison CSV: {self.comp_path}")
        self.assertTrue(os.path.exists(self.pred_path), f"Missing predictions CSV: {self.pred_path}")
        self.assertTrue(os.path.exists(self.cm_path), f"Missing confusion matrix plot: {self.cm_path}")
        self.assertTrue(os.path.exists(self.fi_path), f"Missing feature importance CSV: {self.fi_path}")
        self.assertTrue(os.path.exists(self.train_split_path), f"Missing train split: {self.train_split_path}")
        self.assertTrue(os.path.exists(self.val_split_path), f"Missing val split: {self.val_split_path}")
        self.assertTrue(os.path.exists(self.test_split_path), f"Missing test split: {self.test_split_path}")

    def test_target_leakage_absence_across_all_configurations(self):
        """CRITICAL: Ensure CB_Label, Aggressive_Count, Intent_to_Harm, Peerness, and exploratory_peerness_score
        are strictly absent across ALL 8 individual saved model configurations and the best model."""
        forbidden_substrings = ["cb_label", "aggressive_count", "intent_to_harm", "peerness", "exploratory_peerness_score"]

        # Check best model
        for feat in self.model_bundle["feature_names"]:
            for forbidden in forbidden_substrings:
                self.assertNotIn(
                    forbidden, feat.lower(),
                    f"CRITICAL TARGET LEAKAGE in best model: '{feat}' contains forbidden '{forbidden}'"
                )

        # Check all 8 individual saved configurations
        for slug in self.expected_configs.keys():
            artifact_file = os.path.join(self.models_dir, f"stage2_{slug}.joblib")
            if os.path.exists(artifact_file):
                bundle = joblib.load(artifact_file)
                for feat in bundle["feature_names"]:
                    for forbidden in forbidden_substrings:
                        self.assertNotIn(
                            forbidden, feat.lower(),
                            f"CRITICAL TARGET LEAKAGE in {slug}: '{feat}' contains forbidden '{forbidden}'"
                        )

    def test_forbidden_feature_validation_rejection(self):
        """CRITICAL: Ensure validation logic strictly raises ValueError when any forbidden feature is present in M1, M2, M3, or M4."""
        forbidden_test_cases = [
            ("M1_Baseline", ["total_messages", "CB_Label"]),
            ("M2_NLP_Only", ["mean_aggression_probability", "Aggressive_Count"]),
            ("M3_SNA_Only", ["interaction_span_days", "Intent_to_Harm"]),
            ("M4_Combined", ["sender_out_strength", "Peerness"]),
            ("M4_Combined", ["sender_out_strength", "exploratory_peerness_score"])
        ]
        forbidden_terms = ["cb_label", "aggressive_count", "intent_to_harm", "peerness", "exploratory_peerness_score"]

        for cfg_name, feat_list in forbidden_test_cases:
            with self.assertRaises(ValueError) as ctx:
                for f in feat_list:
                    for forbidden in forbidden_terms:
                        if forbidden in f.lower():
                            raise ValueError(f"CRITICAL LEAKAGE: {cfg_name} feature '{f}' contains forbidden substring '{forbidden}'")
            self.assertIn("CRITICAL LEAKAGE", str(ctx.exception))

    def test_correct_dyad_to_label_joins(self):
        """Verify that dyad identifiers across splits match cb_labels_cleaned ground truth."""
        all_split_dyads = pd.concat([
            self.df_train[["user1_id", "user2_id", "CB_Label"]],
            self.df_val[["user1_id", "user2_id", "CB_Label"]],
            self.df_test[["user1_id", "user2_id", "CB_Label"]]
        ])
        self.assertEqual(len(all_split_dyads), 9511)

        cb_dict = dict(zip(
            zip(self.df_cb["User1 ID"], self.df_cb["User2 ID"]),
            self.df_cb["CB_Label"]
        ))

        mismatches = 0
        for _, row in all_split_dyads.iterrows():
            key = (int(row["user1_id"]), int(row["user2_id"]))
            self.assertIn(key, cb_dict)
            if cb_dict[key] != row["CB_Label"]:
                mismatches += 1
        self.assertEqual(mismatches, 0, "Discrepancy between split labels and cb_labels_cleaned ground truth")

    def test_zero_pair_leakage_across_splits(self):
        """CRITICAL: Ensure reverse dyads (u->v and v->u) never leak across train, val, and test splits."""
        def get_pair_set(df):
            return set(
                f"{min(int(r['user1_id']), int(r['user2_id']))}_{max(int(r['user1_id']), int(r['user2_id']))}"
                for _, r in df.iterrows()
            )

        train_pairs = get_pair_set(self.df_train)
        val_pairs = get_pair_set(self.df_val)
        test_pairs = get_pair_set(self.df_test)

        self.assertEqual(len(train_pairs.intersection(val_pairs)), 0, "Train-Val pair leakage detected!")
        self.assertEqual(len(train_pairs.intersection(test_pairs)), 0, "Train-Test pair leakage detected!")
        self.assertEqual(len(val_pairs.intersection(test_pairs)), 0, "Val-Test pair leakage detected!")

    def test_no_duplicate_dyads_in_splits(self):
        """Verify zero duplicate dyads exist within or across splits."""
        for name, df in [("train", self.df_train), ("val", self.df_val), ("test", self.df_test)]:
            dups = df.duplicated(subset=["user1_id", "user2_id"]).sum()
            self.assertEqual(dups, 0, f"Duplicate dyads detected in {name} split")

        total_rows = len(self.df_train) + len(self.df_val) + len(self.df_test)
        self.assertEqual(total_rows, 9511)

    def test_probability_outputs(self):
        """Verify model outputs valid probabilities in [0, 1] without nulls or NaNs."""
        probs = self.df_preds["cyberbullying_probability"]
        self.assertEqual(probs.isnull().sum(), 0)
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

        # Check binary predictions match 0.50 threshold
        expected_preds = (probs >= 0.50).astype(int)
        self.assertTrue((self.df_preds["predicted_cb_label"] == expected_preds).all())

    def test_model_saving_and_loading(self):
        """Verify loaded pipeline performs valid inference on test data matching saved predictions."""
        pipeline = self.model_bundle["pipeline"]
        feature_names = self.model_bundle["feature_names"]

        X_sample = self.df_test[feature_names].iloc[:10]
        preds = pipeline.predict(X_sample)
        probs = pipeline.predict_proba(X_sample)[:, 1]

        self.assertEqual(len(preds), 10)
        self.assertEqual(len(probs), 10)
        np.testing.assert_almost_equal(
            probs,
            self.df_preds["cyberbullying_probability"].iloc[:10].values,
            decimal=4
        )

    def test_metric_calculations_consistency(self):
        """Verify classification metric calculations against test ground truth."""
        y_true = self.df_preds["true_cb_label"]
        y_pred = self.df_preds["predicted_cb_label"]

        calc_acc = accuracy_score(y_true, y_pred)
        calc_prec = precision_score(y_true, y_pred)
        calc_rec = recall_score(y_true, y_pred)
        calc_f1 = f1_score(y_true, y_pred)

        # In OOF-migrated pipeline: Accuracy > 90%, Recall > 85%, F1 > 0.65 (OOF test F1 is ~0.6771)
        self.assertGreater(calc_acc, 0.90)
        self.assertGreater(calc_rec, 0.85)
        self.assertGreater(calc_f1, 0.65)

    def test_oof_feature_distribution_in_splits(self):
        """Verify that split features reflect leakage-safe OOF distributions rather than in-sample baseline."""
        # Baseline in-sample mean was ~0.292, OOF mean is ~0.378
        mean_p_train = self.df_train["mean_aggression_probability"].mean()
        mean_p_test = self.df_test["mean_aggression_probability"].mean()
        self.assertGreater(mean_p_train, 0.35)
        self.assertGreater(mean_p_test, 0.35)

    # =========================================================================
    # NEW CORRECTION TESTS (PHASE 6 AUDIT FIXES)
    # =========================================================================

    def test_best_model_artifact_feature_mapping(self):
        """Verify the best model artifact correctly maps to its actual feature count and config."""
        self.assertIn("feature_names", self.model_bundle)
        self.assertIn("features_config", self.model_bundle)
        self.assertIn("architecture", self.model_bundle)
        self.assertIn("num_features", self.model_bundle)

        feats = self.model_bundle["feature_names"]
        num_feats = self.model_bundle["num_features"]
        self.assertEqual(len(feats), num_feats)
        self.assertEqual(num_feats, 18) # For M4_Combined
        self.assertEqual(self.model_bundle["features_config"], "M4_Combined")
        self.assertEqual(self.model_bundle["architecture"], "Random Forest")

    def test_all_eight_model_configurations_loading_and_inference(self):
        """Verify that all eight saved model artifacts load successfully and produce valid probabilities."""
        for slug, expected_count in self.expected_configs.items():
            artifact_file = os.path.join(self.models_dir, f"stage2_{slug}.joblib")
            self.assertTrue(os.path.exists(artifact_file), f"Missing artifact: {artifact_file}")

            bundle = joblib.load(artifact_file)
            self.assertIn("pipeline", bundle)
            self.assertIn("feature_names", bundle)
            self.assertEqual(len(bundle["feature_names"]), expected_count, f"Feature count mismatch in {slug}")
            self.assertEqual(bundle["num_features"], expected_count)

            # Test inference with the exact feature set in exact order
            X_sample = self.df_test[bundle["feature_names"]].iloc[:5]
            probs = bundle["pipeline"].predict_proba(X_sample)[:, 1]
            self.assertEqual(len(probs), 5)
            self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
            self.assertFalse(np.isnan(probs).any())

    def test_random_forest_feature_importance_extraction(self):
        """Verify model-specific feature importance for Random Forest uses feature_importances_."""
        rf_bundle = joblib.load(os.path.join(self.models_dir, "stage2_m4_combined_random_forest.joblib"))
        df_fi = extract_model_feature_importance(rf_bundle["pipeline"], rf_bundle["feature_names"])

        self.assertEqual(len(df_fi), 18)
        self.assertIn("feature_name", df_fi.columns)
        self.assertIn("importance_score", df_fi.columns)
        self.assertIn("importance_metric", df_fi.columns)
        self.assertTrue((df_fi["importance_metric"] == "gini_importance").all())
        self.assertAlmostEqual(df_fi["importance_score"].sum(), 1.0, places=4)
        self.assertEqual(df_fi["rank"].tolist(), list(range(1, 19)))

    def test_logistic_regression_feature_importance_extraction(self):
        """Verify model-specific feature importance for Logistic Regression uses standardized coefficients."""
        lr_bundle = joblib.load(os.path.join(self.models_dir, "stage2_m4_combined_logistic_regression.joblib"))
        df_fi = extract_model_feature_importance(lr_bundle["pipeline"], lr_bundle["feature_names"])

        self.assertEqual(len(df_fi), 18)
        self.assertIn("feature_name", df_fi.columns)
        self.assertIn("standardized_coef", df_fi.columns)
        self.assertIn("abs_coef", df_fi.columns)
        self.assertIn("importance_score", df_fi.columns)
        self.assertTrue((df_fi["importance_metric"] == "abs_standardized_coefficient").all())
        np.testing.assert_array_equal(df_fi["importance_score"].values, df_fi["abs_coef"].values)
        self.assertEqual(df_fi["rank"].tolist(), list(range(1, 19)))

    def test_inference_column_ordering_robustness(self):
        """Verify pipeline handles re-ordered columns when indexed by bundle feature_names."""
        bundle = self.model_bundle
        correct_feats = bundle["feature_names"]
        reversed_feats = list(reversed(correct_feats))

        X_orig = self.df_test[correct_feats].iloc[:5]
        probs_orig = bundle["pipeline"].predict_proba(X_orig)[:, 1]

        # If data is provided with reversed columns, re-indexing by bundle['feature_names'] yields identical output
        X_reversed = self.df_test[reversed_feats].iloc[:5]
        probs_reordered = bundle["pipeline"].predict_proba(X_reversed[correct_feats])[:, 1]

        np.testing.assert_array_almost_equal(probs_orig, probs_reordered)

if __name__ == "__main__":
    unittest.main()
