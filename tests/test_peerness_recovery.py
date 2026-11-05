import unittest
import os
import pandas as pd
import numpy as np

class TestPeernessRecovery(unittest.TestCase):
    def setUp(self):
        self.peer_raw = pd.read_csv("datasets/2. peerness_values.csv")
        self.cb_raw = pd.read_csv("datasets/6. CB_Labels.csv")
        self.cb_cleaned = pd.read_csv("data/processed/cb_labels_cleaned.csv")
        self.recovery_report = pd.read_csv("results/peerness_recovery_report.csv")
        self.peer_sym = pd.read_csv("data/processed/peerness_symmetric.csv")

        # Build reference lookup
        self.peer_dict = {}
        for _, row in self.peer_raw.iterrows():
            u1, u2, p = int(row["User 1"]), int(row["User 2"]), float(row["Peerness"])
            self.peer_dict[(u1, u2)] = p
            self.peer_dict[(u2, u1)] = p

    def test_zero_missing_peerness_after_recovery(self):
        """Verify that cb_labels_cleaned has exactly 0 missing/NaN Peerness values."""
        null_count = self.cb_cleaned["Peerness"].isna().sum()
        self.assertEqual(null_count, 0, f"Found {null_count} NaNs remaining in Peerness!")

    def test_recovered_count_exact_4686(self):
        """Verify that exactly 4,686 records were recovered."""
        recovered_count = self.cb_cleaned["peerness_was_recovered"].sum()
        self.assertEqual(recovered_count, 4686, f"Expected 4686 recovered rows, got {recovered_count}")

    def test_all_recovered_scores_match_source_exactly(self):
        """Verify every recovered score strictly matches the source peerness_values.csv."""
        recovered_rows = self.cb_cleaned[self.cb_cleaned["peerness_was_recovered"]]
        for _, row in recovered_rows.iterrows():
            u1, u2, p_rec = int(row["User1 ID"]), int(row["User2 ID"]), float(row["Peerness"])
            p_expected = self.peer_dict[(u1, u2)]
            self.assertTrue(
                np.isclose(p_rec, p_expected),
                f"Mismatch for ({u1}, {u2}): recovered={p_rec}, expected={p_expected}"
            )

    def test_originally_present_scores_not_corrupted(self):
        """Verify originally present scores (4,825 rows) remain unchanged."""
        orig_present_rows = self.cb_cleaned[~self.cb_cleaned["peerness_was_recovered"]]
        self.assertEqual(len(orig_present_rows), 4825)
        for _, row in orig_present_rows.iterrows():
            u1, u2, p_cleaned = int(row["User1 ID"]), int(row["User2 ID"]), float(row["Peerness"])
            p_expected = self.peer_dict[(u1, u2)]
            self.assertTrue(
                np.isclose(p_cleaned, p_expected),
                f"Corrupted original value for ({u1}, {u2})"
            )

    def test_peerness_range_valid(self):
        """Verify all peerness scores fall within documented range (0, 1]."""
        min_p = self.cb_cleaned["Peerness"].min()
        max_p = self.cb_cleaned["Peerness"].max()
        self.assertGreater(min_p, 0.0, "Peerness must not be 0 or negative")
        self.assertLessEqual(max_p, 1.0, "Peerness must not exceed 1.0")
        self.assertTrue(
            (self.cb_cleaned["Peerness"] > 0.09).all() and (self.cb_cleaned["Peerness"] < 1.0).all()
        )

    def test_symmetric_peerness_table_completeness(self):
        """Verify peerness_symmetric.csv has 9900 directed pairs (100 * 99)."""
        self.assertEqual(len(self.peer_sym), 9900)

        # Check symmetry
        for _, row in self.peer_sym.iterrows():
            u1, u2, p = int(row["User 1"]), int(row["User 2"]), float(row["Peerness"])
            self.assertTrue((u1, u2) in self.peer_dict)
            self.assertTrue(
                np.isclose(p, self.peer_dict[(u1, u2)], rtol=1e-9, atol=1e-12),
                f"Peerness mismatch for ({u1}, {u2})"
            )


if __name__ == "__main__":
    unittest.main()

