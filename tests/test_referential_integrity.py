import unittest
import pandas as pd

class TestReferentialIntegrity(unittest.TestCase):
    def setUp(self):
        self.users = pd.read_csv("data/processed/users_cleaned.csv")
        self.peer_sym = pd.read_csv("data/processed/peerness_symmetric.csv")
        self.cb = pd.read_csv("data/processed/cb_labels_cleaned.csv")
        self.comm = pd.read_csv("data/processed/communication_cleaned_sorted.csv")

    def test_user_ids_contiguous_1_to_100(self):
        """Verify 100 users with IDs 1 to 100 and no missing or duplicate IDs."""
        uids = self.users["UserID"].tolist()
        self.assertEqual(len(uids), 100)
        self.assertEqual(len(set(uids)), 100)
        self.assertEqual(sorted(uids), list(range(1, 101)))

    def test_communication_user_ids_valid(self):
        """Verify all senders and receivers belong to valid UserIDs."""
        valid_uids = set(self.users["UserID"])
        comm_senders = set(self.comm["User1 ID"])
        comm_receivers = set(self.comm["User2 ID"])
        self.assertTrue(comm_senders.issubset(valid_uids))
        self.assertTrue(comm_receivers.issubset(valid_uids))

    def test_cb_labels_user_ids_valid(self):
        """Verify all dyad endpoints belong to valid UserIDs."""
        valid_uids = set(self.users["UserID"])
        cb_u1 = set(self.cb["User1 ID"])
        cb_u2 = set(self.cb["User2 ID"])
        self.assertTrue(cb_u1.issubset(valid_uids))
        self.assertTrue(cb_u2.issubset(valid_uids))

    def test_peerness_symmetric_user_ids_valid(self):
        """Verify all pairs in peerness_symmetric belong to valid UserIDs."""
        valid_uids = set(self.users["UserID"])
        p_u1 = set(self.peer_sym["User 1"])
        p_u2 = set(self.peer_sym["User 2"])
        self.assertTrue(p_u1.issubset(valid_uids))
        self.assertTrue(p_u2.issubset(valid_uids))

    def test_bijective_dyad_mapping_comm_and_cb(self):
        """Verify exact 1-to-1 dyad match between communication data and CB labels."""
        comm_pairs = set(zip(self.comm["User1 ID"], self.comm["User2 ID"]))
        cb_pairs = set(zip(self.cb["User1 ID"], self.cb["User2 ID"]))
        self.assertEqual(comm_pairs, cb_pairs)
        self.assertEqual(len(comm_pairs), 9511)

if __name__ == "__main__":
    unittest.main()
