import os
import time
import networkx as nx
import pandas as pd
import numpy as np

def extract_sna_features(
    users_path="data/processed/users_cleaned.csv",
    pred_comm_path="data/processed/communication_with_oof_predictions.csv",
    peerness_path="data/processed/peerness_symmetric.csv",
    results_dir="results",
    processed_dir="data/processed"
):
    print("=" * 70)
    print("CYBERSHIELD AI — STEP 4: SNA FEATURE ENGINEERING (OOF-MIGRATED)")
    print("=" * 70)

    os.makedirs(results_dir, exist_ok=True)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading user nodes from: {users_path}")
    df_users = pd.read_csv(users_path)
    print(f"Loading communication events with OOF predictions from: {pred_comm_path}")
    df_comm = pd.read_csv(pred_comm_path)
    df_peer = pd.read_csv(peerness_path)

    # Validate required columns
    required_cols = ["User1 ID", "User2 ID", "timestamp", "aggression_probability", "aggression_prediction"]
    for col in required_cols:
        if col not in df_comm.columns:
            raise ValueError(f"CRITICAL ERROR: Communication dataset missing required column '{col}'!")

    df_comm["dt"] = pd.to_datetime(df_comm["timestamp"])

    # Build symmetric peerness lookup
    peer_dict = {}
    for _, row in df_peer.iterrows():
        u1, u2, p = int(row["User 1"]), int(row["User 2"]), float(row["Peerness"])
        peer_dict[(u1, u2)] = p

    # 2. Build Aggregated DiGraph for Centrality Calculations
    print("\n>>> Building Aggregated DiGraph for Network Centralities...")
    t0 = time.time()
    G_di = nx.DiGraph()

    # Add 100 user nodes
    for _, row in df_users.iterrows():
        u_id = int(row["UserID"])
        G_di.add_node(
            u_id,
            user_id=u_id,
            age=int(row["Age"]),
            gender=str(row["Gender"]),
            school=str(row["School Name"]),
            grade=int(row["Grade"])
        )

    # Edge aggregation
    dyad_msg_counts = {}
    dyad_agg_probs = {}

    for (u, v), group in df_comm.groupby(["User1 ID", "User2 ID"]):
        u_int, v_int = int(u), int(v)
        total_m = len(group)
        agg_vol = float(group["aggression_probability"].sum())
        mean_p = float(group["aggression_probability"].mean())
        max_p = float(group["aggression_probability"].max())

        dyad_msg_counts[(u_int, v_int)] = total_m
        dyad_agg_probs[(u_int, v_int)] = {
            "total_messages": total_m,
            "mean_prob": mean_p,
            "sum_prob": agg_vol,
            "max_prob": max_p
        }

        # Weight inverted for shortest-path distance in betweenness
        dist_w = 1.0 / (total_m + 1.0)

        G_di.add_edge(
            u_int, v_int,
            weight=total_m,
            agg_weight=agg_vol,
            distance=dist_w
        )

    print(f"Graph constructed in {time.time()-t0:.2f}s | Nodes: {G_di.number_of_nodes()} | Edges: {G_di.number_of_edges()}")

    # 3. Compute Node-Level Centralities
    print("\n>>> Computing Node-Level Graph Centralities...")
    t0 = time.time()

    # PageRank (weighted by message count and by aggression volume)
    pagerank_msg = nx.pagerank(G_di, weight="weight", max_iter=500)
    pagerank_agg = nx.pagerank(G_di, weight="agg_weight", max_iter=500)

    # Betweenness centrality (using distance)
    betweenness = nx.betweenness_centrality(G_di, weight="distance", normalized=True)

    # HITS Hub and Authority
    try:
        hub_scores, auth_scores = nx.hits(G_di, max_iter=1000, normalized=True)
    except Exception as e:
        print(f"Warning: HITS convergence issue ({e}), using default fallback.")
        hub_scores = {n: 0.0 for n in G_di.nodes()}
        auth_scores = {n: 0.0 for n in G_di.nodes()}

    # Clustering coefficient
    clustering = nx.clustering(G_di)

    print(f"Centralities calculated in {time.time()-t0:.2f}s")

    # 4. Assemble Node-Level SNA Features
    print("\n>>> Assembling results/node_sna_features.csv...")
    node_records = []

    for _, row in df_users.iterrows():
        u = int(row["UserID"])

        # Degree & Strength
        in_deg = int(G_di.in_degree(u))
        out_deg = int(G_di.out_degree(u))
        in_str = int(G_di.in_degree(u, weight="weight"))
        out_str = int(G_di.out_degree(u, weight="weight"))
        total_act = in_str + out_str
        act_imbalance = (out_str - in_str) / (total_act + 1e-6)

        # Aggression exposure / transmission
        in_agg_vol = float(G_di.in_degree(u, weight="agg_weight"))
        out_agg_vol = float(G_di.out_degree(u, weight="agg_weight"))
        mean_in_agg = (in_agg_vol / in_str) if in_str > 0 else 0.0
        mean_out_agg = (out_agg_vol / out_str) if out_str > 0 else 0.0
        trans_ratio = out_agg_vol / (in_agg_vol + 1.0)

        node_records.append({
            "user_id": u,
            "age": int(row["Age"]),
            "gender": str(row["Gender"]),
            "school": str(row["School Name"]),
            "grade": int(row["Grade"]),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "in_strength": in_str,
            "out_strength": out_str,
            "total_activity_strength": total_act,
            "activity_imbalance": round(act_imbalance, 6),
            "incoming_aggression_volume": round(in_agg_vol, 6),
            "outgoing_aggression_volume": round(out_agg_vol, 6),
            "mean_incoming_aggression_prob": round(mean_in_agg, 6),
            "mean_outgoing_aggression_prob": round(mean_out_agg, 6),
            "aggression_transmission_ratio": round(trans_ratio, 6),
            "pagerank_message_weighted": round(pagerank_msg[u], 6),
            "pagerank_aggression_weighted": round(pagerank_agg[u], 6),
            "betweenness_centrality": round(betweenness[u], 6),
            "hub_score": round(hub_scores[u], 6),
            "authority_score": round(auth_scores[u], 6),
            "clustering_coefficient": round(clustering[u], 6)
        })

    df_node_features = pd.DataFrame(node_records)
    node_out_path = os.path.join(results_dir, "node_sna_features.csv")
    df_node_features.to_csv(node_out_path, index=False)
    print(f"Saved: {node_out_path} ({len(df_node_features)} rows)")
    if processed_dir:
        node_proc_path = os.path.join(processed_dir, "node_sna_features.csv")
        df_node_features.to_csv(node_proc_path, index=False)
        print(f"Saved: {node_proc_path} ({len(df_node_features)} rows)")

    # Node dictionary for fast dyad lookup
    node_feat_dict = df_node_features.set_index("user_id").to_dict(orient="index")

    # 5. Assemble Dyad-Level SNA Features
    print("\n>>> Assembling results/dyad_sna_features.csv (9,511 communicating pairs)...")
    t0 = time.time()
    dyad_records = []

    # Group communication events by dyad
    comm_dyads = df_comm.groupby(["User1 ID", "User2 ID"])

    for (u1, u2), group in comm_dyads:
        u1_int, u2_int = int(u1), int(u2)

        n_forward = len(group)
        n_reverse = dyad_msg_counts.get((u2_int, u1_int), 0)
        total_dyad_msgs = n_forward + n_reverse

        # Reciprocity and imbalance
        reciprocity_ratio = (n_reverse / total_dyad_msgs) if total_dyad_msgs > 0 else 0.0
        comm_imbalance = (n_forward - n_reverse) / (total_dyad_msgs + 1e-6)

        # Aggression statistics
        mean_p = float(group["aggression_probability"].mean())
        sum_p = float(group["aggression_probability"].sum())
        max_p = float(group["aggression_probability"].max())

        # Temporal dynamics & burstiness
        ts_series = group["dt"].sort_values()
        first_ts = ts_series.iloc[0]
        last_ts = ts_series.iloc[-1]
        span_days = float((last_ts - first_ts).total_seconds() / 86400.0)

        if n_forward > 1:
            diffs_hours = ts_series.diff().dropna().dt.total_seconds() / 3600.0
            mean_inter_hours = float(diffs_hours.mean())
            std_inter_hours = float(diffs_hours.std()) if len(diffs_hours) > 1 else 0.0
            if np.isnan(std_inter_hours):
                std_inter_hours = 0.0
        else:
            mean_inter_hours = 0.0
            std_inter_hours = 0.0

        # Retrieve sender and receiver node centralities
        s_meta = node_feat_dict[u1_int]
        r_meta = node_feat_dict[u2_int]

        # Exploratory peerness (clearly retained for exploratory use only, NOT predictive modeling)
        peerness_val = peer_dict.get((u1_int, u2_int), np.nan)

        dyad_records.append({
            "user1_id": u1_int,
            "user2_id": u2_int,
            "total_messages": int(n_forward),
            "mean_aggression_probability": round(mean_p, 6),
            "predicted_aggression_volume": round(sum_p, 6),
            "max_aggression_probability": round(max_p, 6),
            "reciprocal_messages": int(n_reverse),
            "reciprocity_ratio": round(reciprocity_ratio, 6),
            "directional_communication_imbalance": round(comm_imbalance, 6),
            "interaction_span_days": round(span_days, 4),
            "inter_message_time_mean_hours": round(mean_inter_hours, 4),
            "inter_message_time_std_hours": round(std_inter_hours, 4),
            "sender_out_strength": int(s_meta["out_strength"]),
            "receiver_in_strength": int(r_meta["in_strength"]),
            "sender_outgoing_aggression_volume": round(float(s_meta["outgoing_aggression_volume"]), 6),
            "receiver_incoming_aggression_exposure": round(float(r_meta["incoming_aggression_volume"]), 6),
            "sender_pagerank": round(float(s_meta["pagerank_message_weighted"]), 6),
            "receiver_pagerank": round(float(r_meta["pagerank_message_weighted"]), 6),
            "sender_hub_score": round(float(s_meta["hub_score"]), 6),
            "receiver_authority_score": round(float(r_meta["authority_score"]), 6),
            "exploratory_peerness_score": round(float(peerness_val), 6)
        })

    df_dyad_features = pd.DataFrame(dyad_records)
    dyad_out_path = os.path.join(results_dir, "dyad_sna_features.csv")
    df_dyad_features.to_csv(dyad_out_path, index=False)
    print(f"Saved: {dyad_out_path} ({len(df_dyad_features)} rows) in {time.time()-t0:.2f}s")
    if processed_dir:
        dyad_proc_path = os.path.join(processed_dir, "dyad_sna_features.csv")
        df_dyad_features.to_csv(dyad_proc_path, index=False)
        print(f"Saved: {dyad_proc_path} ({len(df_dyad_features)} rows)")

    # 6. Build SNA Summary Table
    print("\n>>> Building results/sna_summary.csv...")
    summary_records = [
        {"metric_name": "total_registered_users", "value": len(df_users), "description": "Total students in the cohort (nodes in graph)"},
        {"metric_name": "total_communication_events", "value": len(df_comm), "description": "Total messages exchanged (edges in MultiDiGraph)"},
        {"metric_name": "total_directed_dyads", "value": len(df_dyad_features), "description": "Unique ordered communicating pairs (edges in DiGraph)"},
        {"metric_name": "network_density", "value": round(nx.density(G_di), 4), "description": "Density of directed communication graph"},
        {"metric_name": "mean_out_degree", "value": round(float(df_node_features["out_degree"].mean()), 2), "description": "Mean number of distinct recipients per user"},
        {"metric_name": "mean_in_degree", "value": round(float(df_node_features["in_degree"].mean()), 2), "description": "Mean number of distinct senders per user"},
        {"metric_name": "mean_messages_per_dyad", "value": round(float(df_dyad_features["total_messages"].mean()), 2), "description": "Mean messages transmitted per ordered pair"},
        {"metric_name": "mean_predicted_aggression_prob", "value": round(float(df_comm["aggression_probability"].mean()), 4), "description": "Mean calibrated aggression probability across messages"},
        {"metric_name": "total_predicted_aggressive_messages", "value": int((df_comm["aggression_prediction"] == 1).sum()), "description": "Count of messages classified as aggressive (P >= 0.5)"},
        {"metric_name": "reciprocal_dyads_count", "value": int((df_dyad_features["reciprocal_messages"] > 0).sum()), "description": "Ordered dyads where reverse communication exists"},
        {"metric_name": "asymmetric_dyads_count", "value": int((df_dyad_features["reciprocal_messages"] == 0).sum()), "description": "Ordered dyads with strictly one-way communication"},
        {"metric_name": "max_outgoing_aggression_volume_user", "value": int(df_node_features.loc[df_node_features["outgoing_aggression_volume"].idxmax(), "user_id"]), "description": "User with highest total outgoing predicted aggression volume"},
        {"metric_name": "max_incoming_aggression_volume_user", "value": int(df_node_features.loc[df_node_features["incoming_aggression_volume"].idxmax(), "user_id"]), "description": "User with highest total incoming predicted aggression exposure"}
    ]

    df_sna_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(results_dir, "sna_summary.csv")
    df_sna_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print("=" * 70)
    print("SNA FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return df_node_features, df_dyad_features, df_sna_summary

if __name__ == "__main__":
    extract_sna_features()
