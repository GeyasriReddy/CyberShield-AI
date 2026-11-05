import unittest
import pandas as pd

class TestChronologicalSorting(unittest.TestCase):
    def setUp(self):
        self.comm_raw = pd.read_csv("datasets/5. Communication_Data_Among_Users.csv")
        self.comm_sorted = pd.read_csv("data/processed/communication_cleaned_sorted.csv")
        self.cb_cleaned = pd.read_csv("data/processed/cb_labels_cleaned.csv")

    def test_total_row_count_preserved(self):
        """Verify no communication records were deleted or created."""
        self.assertEqual(len(self.comm_sorted), len(self.comm_raw))
        self.assertEqual(len(self.comm_sorted), 90356)

    def test_strictly_chronological_ordering_per_pair(self):
        """Verify timestamps are monotonically non-decreasing for 100% of communicating pairs."""
        # Check monotonicity per pair
        is_monotonic = self.comm_sorted.groupby(["User1 ID", "User2 ID"])["timestamp"].apply(
            lambda s: s.is_monotonic_increasing
        )
        self.assertEqual(len(is_monotonic), 9511)
        self.assertTrue(
            is_monotonic.all(),
            f"Failed chronological sorting for {(~is_monotonic).sum()} pairs!"
        )

    def test_timestamp_integrity(self):
        """Verify timestamp matches original Date and Time."""
        expected_ts = self.comm_sorted["Date"] + " " + self.comm_sorted["Time"] + ":00"
        actual_ts = self.comm_sorted["timestamp"].astype(str)
        self.assertTrue((actual_ts == expected_ts).all())

    def test_directional_message_counts_preserved(self):
        """Verify Total_messages per pair matches CB_Labels."""
        comm_counts = self.comm_sorted.groupby(["User1 ID", "User2 ID"]).size().reset_index(name="count")
        merged = pd.merge(comm_counts, self.cb_cleaned, on=["User1 ID", "User2 ID"])
        self.assertEqual(len(merged), 9511)
        self.assertTrue((merged["count"] == merged["Total_messages"]).all())

    def test_directional_aggression_counts_preserved(self):
        """Verify Aggressive_Count per pair matches CB_Labels."""
        comm_agg = self.comm_sorted.groupby(["User1 ID", "User2 ID"])["Label"].sum().reset_index(name="agg_count")
        merged = pd.merge(comm_agg, self.cb_cleaned, on=["User1 ID", "User2 ID"])
        self.assertEqual(len(merged), 9511)
        self.assertTrue((merged["agg_count"] == merged["Aggressive_Count"]).all())

if __name__ == "__main__":
    unittest.main()
