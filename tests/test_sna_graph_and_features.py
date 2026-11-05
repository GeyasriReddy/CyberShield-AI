import os
import sys
import unittest
import networkx as nx
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestSNAGraphAndFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.multidigraph_path = os.path.join(PROJECT_ROOT, "results", "communication_multidigraph.graphml")
        cls.digraph_path = os.path.join(PROJECT_ROOT, "results", "communication_digraph.graphml")
        cls.node_features_path = os.path.join(PROJECT_ROOT, "results", "node_sna_features.csv")
        cls.dyad_features_path = os.path.join(PROJECT_ROOT, "results", "dyad_sna_features.csv")
        cls.pred_comm_path = os.path.join(PROJECT_ROOT, "data", "processed", "communication_with_oof_predictions.csv")
        cls.baseline_comm_path = os.path.join(PROJECT_ROOT, "data", "processed", "communication_with_predictions.csv")
        cls.sna_summary_path = os.path.join(PROJECT_ROOT, "results", "sna_summary.csv")
        cls.users_path = os.path.join(PROJECT_ROOT, "data", "processed", "users_cleaned.csv")

        # Load graphs
        cls.G_multi = nx.read_graphml(cls.multidigraph_path)
        cls.G_di = nx.read_graphml(cls.digraph_path)

        # Load tabular datasets
        cls.df_nodes = pd.read_csv(cls.node_features_path)
        cls.df_dyads = pd.read_csv(cls.dyad_features_path)
        cls.df_comm = pd.read_csv(cls.pred_comm_path)
        cls.df_summary = pd.read_csv(cls.sna_summary_path)
        cls.df_users = pd.read_csv(cls.users_path)

    def test_file_existence(self):
        """Verify all graph artifacts, predictions, and feature CSVs exist on disk."""
        self.assertTrue(os.path.exists(self.multidigraph_path))
        self.assertTrue(os.path.exists(self.digraph_path))
        self.assertTrue(os.path.exists(self.node_features_path))
        self.assertTrue(os.path.exists(self.dyad_features_path))
        self.assertTrue(os.path.exists(self.pred_comm_path))
        self.assertTrue(os.path.exists(self.baseline_comm_path))
        self.assertTrue(os.path.exists(self.sna_summary_path))

    def test_node_counts_and_demographics(self):
        """Verify graphs and node features contain exactly 100 nodes matching registered users."""
        self.assertEqual(self.G_multi.number_of_nodes(), 100)
        self.assertEqual(self.G_di.number_of_nodes(), 100)
        self.assertEqual(len(self.df_nodes), 100)

        # Check all user IDs 1 to 100 exist
        expected_ids = set(range(1, 101))
        loaded_node_ids = set(int(n) for n in self.G_di.nodes())
        self.assertEqual(loaded_node_ids, expected_ids)

        # Check demographic attributes on graph nodes
        for node_id, data in self.G_di.nodes(data=True):
            self.assertIn("age", data)
            self.assertIn("gender", data)
            self.assertIn("school", data)
            self.assertIn("grade", data)

    def test_edge_counts_and_self_loops(self):
        """Verify edge counts match specifications and no self-loops exist."""
        # MultiDiGraph: 90,356 directed edges
        self.assertEqual(self.G_multi.number_of_edges(), 90356)
        self.assertEqual(len(self.df_comm), 90356)

        # DiGraph: 9,511 directed dyads
        self.assertEqual(self.G_di.number_of_edges(), 9511)
        self.assertEqual(len(self.df_dyads), 9511)

        # Zero self-loops
        self.assertEqual(nx.number_of_selfloops(self.G_multi), 0)
        self.assertEqual(nx.number_of_selfloops(self.G_di), 0)

    def test_prediction_integrity_in_communication(self):
        """Verify communication predictions have no missing values and probabilities are strictly in [0, 1]."""
        self.assertEqual(self.df_comm["aggression_probability"].isnull().sum(), 0)
        self.assertEqual(self.df_comm["aggression_prediction"].isnull().sum(), 0)

        min_prob = self.df_comm["aggression_probability"].min()
        max_prob = self.df_comm["aggression_probability"].max()
        self.assertGreaterEqual(min_prob, 0.0)
        self.assertLessEqual(max_prob, 1.0)

        # Binary predictions are strictly 0 or 1
        unique_preds = set(self.df_comm["aggression_prediction"].unique())
        self.assertTrue(unique_preds.issubset({0, 1}))

        # Aggression predictions align with threshold 0.5
        threshold_checks = (self.df_comm["aggression_probability"] >= 0.5).astype(int)
        self.assertTrue((self.df_comm["aggression_prediction"] == threshold_checks).all())

    def test_target_leakage_prevention(self):
        """CRITICAL: Ensure CB_Label, Aggressive_Count, and Intent_to_Harm are NEVER present in feature sets."""
        forbidden_columns = [
            "CB_Label", "cb_label", "CB_LABEL",
            "Aggressive_Count", "aggressive_count",
            "Intent_to_Harm", "intent_to_harm"
        ]

        for col in forbidden_columns:
            self.assertNotIn(col, self.df_nodes.columns, f"Target leakage in node features: {col}")
            self.assertNotIn(col, self.df_dyads.columns, f"Target leakage in dyad features: {col}")

        # Ensure Peerness is not present in node features
        self.assertNotIn("peerness", self.df_nodes.columns)
        self.assertNotIn("Peerness", self.df_nodes.columns)

        # In dyad features, peerness is strictly labeled as exploratory
        if "exploratory_peerness_score" in self.df_dyads.columns:
            self.assertNotIn("peerness", self.df_dyads.columns)
            self.assertNotIn("Peerness", self.df_dyads.columns)

    def test_node_feature_completeness_and_validity(self):
        """Verify all essential node-level SNA metrics exist, have no nulls, and are physically valid."""
        required_node_cols = [
            "user_id", "in_degree", "out_degree", "in_strength", "out_strength",
            "total_activity_strength", "activity_imbalance",
            "incoming_aggression_volume", "outgoing_aggression_volume",
            "mean_incoming_aggression_prob", "mean_outgoing_aggression_prob",
            "aggression_transmission_ratio", "pagerank_message_weighted",
            "pagerank_aggression_weighted", "betweenness_centrality",
            "hub_score", "authority_score", "clustering_coefficient"
        ]
        for col in required_node_cols:
            self.assertIn(col, self.df_nodes.columns)
            self.assertEqual(self.df_nodes[col].isnull().sum(), 0)

        # PageRank sum should be approximately 1.0
        self.assertAlmostEqual(self.df_nodes["pagerank_message_weighted"].sum(), 1.0, places=3)
        self.assertAlmostEqual(self.df_nodes["pagerank_aggression_weighted"].sum(), 1.0, places=3)

        # Degree bounds (0 to 99 in a 100-node network)
        self.assertTrue((self.df_nodes["in_degree"] >= 0).all() and (self.df_nodes["in_degree"] <= 99).all())
        self.assertTrue((self.df_nodes["out_degree"] >= 0).all() and (self.df_nodes["out_degree"] <= 99).all())

    def test_dyad_feature_completeness_and_validity(self):
        """Verify all essential dyad-level SNA metrics exist and are mathematically sound."""
        required_dyad_cols = [
            "user1_id", "user2_id", "total_messages", "mean_aggression_probability",
            "predicted_aggression_volume", "max_aggression_probability",
            "reciprocal_messages", "reciprocity_ratio",
            "directional_communication_imbalance", "interaction_span_days",
            "inter_message_time_mean_hours", "inter_message_time_std_hours",
            "sender_out_strength", "receiver_in_strength",
            "sender_outgoing_aggression_volume", "receiver_incoming_aggression_exposure",
            "sender_pagerank", "receiver_pagerank",
            "sender_hub_score", "receiver_authority_score",
            "exploratory_peerness_score"
        ]
        for col in required_dyad_cols:
            self.assertIn(col, self.df_dyads.columns)
            self.assertEqual(self.df_dyads[col].isnull().sum(), 0)

        # Imbalance must be strictly between -1.0 and 1.0
        self.assertTrue((self.df_dyads["directional_communication_imbalance"] >= -1.0).all())
        self.assertTrue((self.df_dyads["directional_communication_imbalance"] <= 1.0).all())

        # Total messages >= 1
        self.assertTrue((self.df_dyads["total_messages"] >= 1).all())

    def test_summary_metrics_consistency(self):
        """Verify summary metrics match graph calculations."""
        summary_dict = dict(zip(self.df_summary["metric_name"], self.df_summary["value"]))

        self.assertEqual(int(summary_dict["total_registered_users"]), 100)
        self.assertEqual(int(summary_dict["total_communication_events"]), 90356)
        self.assertEqual(int(summary_dict["total_directed_dyads"]), 9511)
        self.assertAlmostEqual(float(summary_dict["network_density"]), 0.9607, places=4)
        self.assertEqual(int(summary_dict["total_predicted_aggressive_messages"]), 32504)

    def test_oof_prediction_sources_and_integrity(self):
        """Verify OOF predictions contain all expected prediction source folds and ensemble allocations."""
        self.assertIn("prediction_source", self.df_comm.columns)
        self.assertIn("oof_aggression_probability", self.df_comm.columns)
        sources = set(self.df_comm["prediction_source"].unique())
        expected_sources = {
            "oof_fold_1_prediction", "oof_fold_2_prediction", "oof_fold_3_prediction",
            "oof_fold_4_prediction", "oof_fold_5_prediction",
            "oof_unseen_ensemble_prediction", "oof_conflict_ensemble_prediction"
        }
        self.assertEqual(sources, expected_sources)
        self.assertEqual(self.df_comm["prediction_source"].isnull().sum(), 0)

if __name__ == "__main__":
    unittest.main()
