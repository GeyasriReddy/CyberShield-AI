import unittest
import os
import pandas as pd
import numpy as np
from src.ml_data_prep import cybershield_url_normalizer, simple_norm

class TestMLDataPrep(unittest.TestCase):
    def setUp(self):
        self.conflicts_report = pd.read_csv("results/normalized_label_conflicts.csv")
        self.conflicts_excluded = pd.read_csv("data/processed/ml_corpus_conflict_excluded.csv")
        self.train_split = pd.read_csv("data/processed/ml_corpus_train_split.csv")
        self.val_split = pd.read_csv("data/processed/ml_corpus_val_split.csv")
        self.test_split = pd.read_csv("data/processed/ml_corpus_test_split.csv")
        self.clean_pool = pd.read_csv("data/processed/text_corpus_clean_for_training.csv")

    def test_normalized_label_conflicts_isolated(self):
        """Verify exactly 12 conflicting normalized texts and 24 rows are excluded from splits."""
        self.assertEqual(len(self.conflicts_report), 12)
        self.assertEqual(len(self.conflicts_excluded), 24)
        
        excluded_norms = set(self.conflicts_report["normalized_text"])
        
        train_norms = set(self.train_split["norm_message"])
        val_norms = set(self.val_split["norm_message"])
        test_norms = set(self.test_split["norm_message"])

        self.assertEqual(len(train_norms.intersection(excluded_norms)), 0)
        self.assertEqual(len(val_norms.intersection(excluded_norms)), 0)
        self.assertEqual(len(test_norms.intersection(excluded_norms)), 0)

    def test_url_normalizer_behavior(self):
        """Verify URL normalizer handles raw and stripped URLs and preserves ordinary words & negations."""
        # 1. Standard URL
        t1 = "Check this site http://example.com/test immediately"
        self.assertEqual(cybershield_url_normalizer(t1), "check this site <url> immediately")

        # 2. Punctuation-stripped URL
        t2 = "Go to httpstcojXPDgugY and see"
        self.assertEqual(cybershield_url_normalizer(t2), "go to <url> and see")

        t3 = "Read httpsenwikipediaorgwikiprimespirals now"
        self.assertEqual(cybershield_url_normalizer(t3), "read <url> now")

        # 3. Ordinary words preserved
        t4 = "This is not an attack, I will never hurt anyone"
        self.assertEqual(cybershield_url_normalizer(t4), "this is not an attack, i will never hurt anyone")

        # 4. Negations preserved
        t5 = "Don't you dare do that, neither him nor her"
        res5 = cybershield_url_normalizer(t5)
        for neg in ["don't", "neither", "nor"]:
            self.assertIn(neg, res5)

    def test_frequency_reconciliation(self):
        """Verify exact mathematical reconciliation of 237,585 - 10,403 = 227,182."""
        agg_clean = pd.read_csv("data/processed/text_aggressive_cleaned.csv")
        non_clean = pd.read_csv("data/processed/text_non_aggressive_cleaned.csv")
        ambig = pd.read_csv("data/processed/text_corpus_ambiguous_isolated.csv")

        total_cleaned = len(agg_clean) + len(non_clean)
        self.assertEqual(total_cleaned, 237585)

        agg_ambig = agg_clean["is_ambiguous_overlap"].sum()
        non_ambig = non_clean["is_ambiguous_overlap"].sum()
        self.assertEqual(agg_ambig, 7241)
        self.assertEqual(non_ambig, 3162)
        total_ambig_instances = agg_ambig + non_ambig
        self.assertEqual(total_ambig_instances, 10403)

        expected_unambiguous = total_cleaned - total_ambig_instances
        self.assertEqual(expected_unambiguous, 227182)

        actual_freq_sum = self.clean_pool["frequency"].sum()
        self.assertEqual(actual_freq_sum, 227182)

    def test_split_proportions_and_sizes(self):
        """Verify split sizes sum to 128,414 and respect 70/15/15 target."""
        total_split_rows = len(self.train_split) + len(self.val_split) + len(self.test_split)
        self.assertEqual(total_split_rows, 128414)
        self.assertEqual(len(self.train_split), 89885) # ~70.00%
        self.assertEqual(len(self.val_split), 19258)   # ~15.00%
        self.assertEqual(len(self.test_split), 19271)  # ~15.01%

    def test_zero_normalized_message_overlap_between_splits(self):
        """Verify zero normalized message overlap across Train, Val, and Test."""
        s_train = set(self.train_split["norm_message"])
        s_val = set(self.val_split["norm_message"])
        s_test = set(self.test_split["norm_message"])

        self.assertEqual(len(s_train.intersection(s_val)), 0, "Train-Val normalized overlap found!")
        self.assertEqual(len(s_train.intersection(s_test)), 0, "Train-Test normalized overlap found!")
        self.assertEqual(len(s_val.intersection(s_test)), 0, "Val-Test normalized overlap found!")

    def test_stratified_label_ratios_preserved(self):
        """Verify aggressive ratio is ~26.95% across Train, Val, and Test."""
        for split_df, name in [(self.train_split, "Train"), (self.val_split, "Val"), (self.test_split, "Test")]:
            agg_ratio = (split_df["label"] == 1).mean()
            self.assertAlmostEqual(agg_ratio, 0.2695, delta=0.005, msg=f"{name} ratio drifted!")

if __name__ == "__main__":
    unittest.main()
