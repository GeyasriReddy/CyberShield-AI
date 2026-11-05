import os
import time
import networkx as nx
import pandas as pd
import numpy as np

def build_and_export_graphs(
    users_path="data/processed/users_cleaned.csv",
    pred_comm_path="data/processed/communication_with_oof_predictions.csv",
    results_dir="results"
):
    print("=" * 70)
    print("CYBERSHIELD AI — STEP 3: NETWORKX GRAPH CONSTRUCTION (OOF-MIGRATED)")
    print("=" * 70)

    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading user nodes from: {users_path}")
    df_users = pd.read_csv(users_path)
    print(f"Loading communication events with predictions from: {pred_comm_path}")
    df_comm = pd.read_csv(pred_comm_path)

    # Validate required columns
    required_cols = ["User1 ID", "User2 ID", "timestamp", "Date", "Time", "Label", "aggression_probability", "aggression_prediction", "prediction_status"]
    for col in required_cols:
        if col not in df_comm.columns:
            raise ValueError(f"CRITICAL ERROR: Communication dataset missing required column '{col}'!")

    # 2. Build Graph A: Temporal MultiDiGraph
    print("\n>>> Constructing Graph A: Temporal MultiDiGraph (nx.MultiDiGraph)...")
    t0 = time.time()
    G_multi = nx.MultiDiGraph()

    # Add 100 user nodes with demographic attributes
    for _, row in df_users.iterrows():
        u_id = int(row["UserID"])
        G_multi.add_node(
            u_id,
            user_id=u_id,
            age=int(row["Age"]),
            gender=str(row["Gender"]),
            school=str(row["School Name"]),
            grade=int(row["Grade"])
        )

    # Add 90,356 directed edges (one per communication event)
    for idx, row in df_comm.iterrows():
        u = int(row["User1 ID"])
        v = int(row["User2 ID"])
        if u == v:
            continue # Ensure strictly no self-loops

        edge_attrs = {
            "message_id": int(idx),
            "timestamp": str(row["timestamp"]),
            "date": str(row["Date"]),
            "time": str(row["Time"]),
            "aggression_probability": float(row["aggression_probability"]),
            "aggression_prediction": int(row["aggression_prediction"]),
            "ground_truth_label": int(row["Label"]),
            "is_ambiguous_overlap": bool(row.get("is_ambiguous_overlap", False)),
            "prediction_status": str(row["prediction_status"])
        }
        if "oof_aggression_probability" in row:
            edge_attrs["oof_aggression_probability"] = float(row["oof_aggression_probability"])
        if "prediction_source" in row:
            edge_attrs["prediction_source"] = str(row["prediction_source"])

        G_multi.add_edge(u, v, **edge_attrs)

    print(f"Graph A constructed in {time.time()-t0:.2f}s:")
    print(f"  Nodes: {G_multi.number_of_nodes():,} | Edges: {G_multi.number_of_edges():,}")
    print(f"  Self loops: {nx.number_of_selfloops(G_multi)}")

    # Export Graph A
    multidigraph_path = os.path.join(results_dir, "communication_multidigraph.graphml")
    nx.write_graphml(G_multi, multidigraph_path)
    print(f"Saved: {multidigraph_path}")

    # 3. Build Graph B: Aggregated DiGraph
    print("\n>>> Constructing Graph B: Aggregated DiGraph (nx.DiGraph)...")
    t0 = time.time()
    G_di = nx.DiGraph()

    # Add 100 user nodes with demographic attributes
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

    # Convert timestamp to datetime for duration calculations
    df_comm["dt"] = pd.to_datetime(df_comm["timestamp"])

    # Aggregate communication events per directed dyad (User1 ID -> User2 ID)
    dyad_groups = df_comm.groupby(["User1 ID", "User2 ID"])

    for (u, v), group in dyad_groups:
        if u == v:
            continue # No self loops

        total_msgs = len(group)
        mean_prob = float(group["aggression_probability"].mean())
        sum_prob = float(group["aggression_probability"].sum())
        max_prob = float(group["aggression_probability"].max())
        gt_agg = int(group["Label"].sum())

        first_ts = group["dt"].min()
        last_ts = group["dt"].max()
        span_days = float((last_ts - first_ts).total_seconds() / 86400.0)

        G_di.add_edge(
            int(u), int(v),
            total_messages=int(total_msgs),
            mean_aggression_probability=round(mean_prob, 6),
            predicted_aggression_volume=round(sum_prob, 6),
            max_aggression_probability=round(max_prob, 6),
            ground_truth_aggressive_count=int(gt_agg),
            first_interaction_timestamp=str(first_ts),
            last_interaction_timestamp=str(last_ts),
            interaction_span_days=round(span_days, 4)
        )

    print(f"Graph B constructed in {time.time()-t0:.2f}s:")
    print(f"  Nodes: {G_di.number_of_nodes():,} | Edges: {G_di.number_of_edges():,}")
    print(f"  Self loops: {nx.number_of_selfloops(G_di)}")
    print(f"  Network density: {nx.density(G_di):.4f}")

    # Export Graph B
    digraph_path = os.path.join(results_dir, "communication_digraph.graphml")
    nx.write_graphml(G_di, digraph_path)
    print(f"Saved: {digraph_path}")
    print("=" * 70)

    return G_multi, G_di

if __name__ == "__main__":
    build_and_export_graphs()
