import unittest
import pandas as pd

class TestLabelPreservation(unittest.TestCase):
    def setUp(self):
        self.comm_raw = pd.read_csv("datasets/5. Communication_Data_Among_Users.csv")
        self.comm_cleaned = pd.read_csv("data/processed/communication_cleaned_sorted.csv")
        self.cb_raw = pd.read_csv("datasets/6. CB_Labels.csv")
        self.cb_cleaned = pd.read_csv("data/processed/cb_labels_cleaned.csv")
        self.agg_cleaned = pd.read_csv("data/processed/text_aggressive_cleaned.csv")
        self.non_cleaned = pd.read_csv("data/processed/text_non_aggressive_cleaned.csv")
        self.train_clean = pd.read_csv("data/processed/text_corpus_clean_for_training.csv")

    def test_communication_label_distribution_preserved(self):
        """Verify 0/1 label distribution in communication data is completely unchanged."""
        raw_dist = self.comm_raw["Label"].value_counts().to_dict()
        cleaned_dist = self.comm_cleaned["Label"].value_counts().to_dict()
        self.assertEqual(raw_dist, cleaned_dist)
        self.assertEqual(cleaned_dist[0], 59579)
        self.assertEqual(cleaned_dist[1], 30777)

    def test_cb_label_distribution_preserved(self):
        """Verify CB_Label distribution is completely unchanged."""
        raw_cb_dist = self.cb_raw["CB_Label"].value_counts().to_dict()
        cleaned_cb_dist = self.cb_cleaned["CB_Label"].value_counts().to_dict()
        self.assertEqual(raw_cb_dist, cleaned_cb_dist)
        self.assertEqual(cleaned_cb_dist[0], 8519)
        self.assertEqual(cleaned_cb_dist[1], 992)

    def test_source_text_labels_preserved(self):
        """Verify aggressive text has label 1 and non-aggressive text has label 0."""
        self.assertTrue((self.agg_cleaned["source_label"] == 1).all())
        self.assertTrue((self.non_cleaned["source_label"] == 0).all())

    def test_training_pool_labels_valid(self):
        """Verify clean training pool only contains binary labels 0 and 1."""
        unique_labels = set(self.train_clean["label"].unique())
        self.assertEqual(unique_labels, {0, 1})
        self.assertEqual((self.train_clean["label"] == 1).sum(), 34613)
        self.assertEqual((self.train_clean["label"] == 0).sum(), 93825)

if __name__ == "__main__":
    unittest.main()
