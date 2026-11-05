import unittest
import pandas as pd

class TestDuplicateHandling(unittest.TestCase):
    def setUp(self):
        self.comm_cleaned = pd.read_csv("data/processed/communication_cleaned_sorted.csv")
        self.agg_cleaned = pd.read_csv("data/processed/text_aggressive_cleaned.csv")
        self.non_cleaned = pd.read_csv("data/processed/text_non_aggressive_cleaned.csv")
        self.overlap_report = pd.read_csv("results/overlapping_messages_report.csv")
        self.train_clean = pd.read_csv("data/processed/text_corpus_clean_for_training.csv")
        self.ambig_isolated = pd.read_csv("data/processed/text_corpus_ambiguous_isolated.csv")

    def test_no_communication_interactions_deleted(self):
        """Verify that all 90,356 communication events remain intact."""
        self.assertEqual(len(self.comm_cleaned), 90356)

    def test_corpus_frequency_tracked(self):
        """Verify frequency information is preserved in cleaned text sets."""
        self.assertTrue("corpus_frequency" in self.agg_cleaned.columns)
        self.assertTrue("corpus_frequency" in self.non_cleaned.columns)
        self.assertTrue((self.agg_cleaned["corpus_frequency"] >= 1).all())
        self.assertTrue((self.non_cleaned["corpus_frequency"] >= 1).all())

    def test_overlapping_messages_count_exact(self):
        """Verify exactly 2,756 overlapping messages were isolated."""
        self.assertEqual(len(self.overlap_report), 2756)
        self.assertEqual(len(self.ambig_isolated), 2756)

    def test_training_pool_strictly_excludes_overlapping_messages(self):
        """Verify 0 ambiguous overlapping messages enter the clean supervised training pool."""
        overlap_set = set(self.overlap_report["message"])
        train_set = set(self.train_clean["message"])
        intersection = train_set.intersection(overlap_set)
        self.assertEqual(len(intersection), 0, f"Found {len(intersection)} ambiguous messages in clean training pool!")

    def test_training_pool_is_deduplicated(self):
        """Verify clean training pool has distinct messages without duplicates."""
        dup_count = self.train_clean["message"].duplicated().sum()
        self.assertEqual(dup_count, 0)
        self.assertEqual(len(self.train_clean), 128438)

if __name__ == "__main__":
    unittest.main()
