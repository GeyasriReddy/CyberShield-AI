"""
Social Network Analysis (SNA) Module
------------------------------------
Handles graph construction, centrality metrics, community detection,
and academic-grade interactive visualizations using NetworkX and Plotly.
Supports both the 100-user Mendeley cyberbullying communication dataset
and an isolated synthetic fallback graph.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Path to the real Mendeley dataset in workspace
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "A Comprehensive Dataset for Automated Cyberbullying Detection",
    "A Comprehensive Dataset for Automated Cyberbullying Detection"
)
CB_LABELS_PATH = os.path.join(DATASET_DIR, "6. CB_Labels.csv")
COMM_PATH = os.path.join(DATASET_DIR, "5. Communication_Data_Among_Users.csv")


def load_dataset_interaction_graph() -> Tuple[nx.DiGraph, pd.DataFrame, Dict[str, Any]]:
    """
    Constructs a directed NetworkX graph (DiGraph) from the real Mendeley
    cyberbullying communication dataset (100 users, 9,511 dyads, 90,356 messages).
    Falls back to synthetic benchmark if the dataset file is not accessible.
    """
    if os.path.exists(CB_LABELS_PATH):
        try:
            df_cb = pd.read_csv(CB_LABELS_PATH)
            G = nx.DiGraph()

            for _, r in df_cb.iterrows():
                u = f"usr_{int(r['User1 ID']):03d}"
                v = f"usr_{int(r['User2 ID']):03d}"
                G.add_edge(
                    u, v,
                    total_messages=int(r['Total_messages']),
                    aggressive_count=int(r['Aggressive_Count']),
                    intent_to_harm=float(r['Intent_to_Harm']),
                    peerness=float(r['Peerness']),
                    cb_label=int(r['CB_Label'])
                )

            summary = {
                "source": "Mendeley Real Dataset (100 Users)",
                "total_nodes": len(G.nodes),
                "total_edges": len(G.edges),
                "total_messages": int(df_cb['Total_messages'].sum()),
                "total_aggressive": int(df_cb['Aggressive_Count'].sum()),
                "cb_dyads": int((df_cb['CB_Label'] == 1).sum()),
                "benign_dyads": int((df_cb['CB_Label'] == 0).sum()),
                "density": nx.density(G),
                "reciprocity": nx.reciprocity(G),
                "is_real": True
            }
            return G, df_cb, summary
        except Exception as e:
            pass

    # Fallback synthetic graph
    G_synth = load_synthetic_interaction_graph()
    summary = {
        "source": "Synthetic Benchmark (20 Nodes)",
        "total_nodes": len(G_synth.nodes),
        "total_edges": len(G_synth.edges),
        "total_messages": 120,
        "total_aggressive": 18,
        "cb_dyads": 4,
        "benign_dyads": len(G_synth.edges) - 4,
        "density": nx.density(G_synth),
        "reciprocity": 1.0,
        "is_real": False
    }
    return G_synth, pd.DataFrame(), summary


def load_synthetic_interaction_graph() -> nx.Graph:
    """
    Controlled 20-node fallback synthetic network.
    """
    G = nx.Graph()
    c1 = [("usr_101", "usr_102"), ("usr_101", "usr_103"), ("usr_101", "usr_104"),
          ("usr_102", "usr_103"), ("usr_103", "usr_105"), ("usr_104", "usr_106")]
    c2 = [("usr_107", "usr_108"), ("usr_107", "usr_109"), ("usr_108", "usr_109"),
          ("usr_109", "usr_110"), ("usr_110", "usr_111"), ("usr_111", "usr_112")]
    c3 = [("usr_114", "usr_115"), ("usr_115", "usr_116"), ("usr_116", "usr_117"),
          ("usr_117", "usr_118"), ("usr_118", "usr_119"), ("usr_119", "usr_120")]
    bridges = [("usr_103", "usr_107"), ("usr_106", "usr_114"), ("usr_110", "usr_117")]

    for u, v in c1 + c2 + c3 + bridges:
        G.add_edge(u, v, total_messages=10, aggressive_count=1, cb_label=0)

    return G


def compute_network_metrics(G: nx.Graph) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Computes degree centrality, betweenness centrality, in/out degrees, and communities.
    """
    if len(G.nodes) == 0:
        return {}, pd.DataFrame()

    is_directed = G.is_directed()

    # Undirected projection for modularity community detection
    G_undir = G.to_undirected() if is_directed else G
    communities = list(nx.community.greedy_modularity_communities(G_undir))
    community_map = {}
    for idx, comm in enumerate(communities, start=1):
        for node in comm:
            community_map[node] = f"Cluster {idx}"

    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

    metrics_map: Dict[str, Dict[str, Any]] = {}
    records = []

    for node in G.nodes():
        neighbors = list(G.successors(node)) if is_directed else list(G.neighbors(node))
        in_deg = G.in_degree(node) if is_directed else degree_dict.get(node, 0)
        out_deg = G.out_degree(node) if is_directed else degree_dict.get(node, 0)

        # Aggregate aggression sent if edge attributes exist
        aggr_sent = 0
        total_msgs = 0
        if is_directed:
            for _, v, data in G.out_edges(node, data=True):
                aggr_sent += data.get("aggressive_count", 0)
                total_msgs += data.get("total_messages", 0)

        node_info = {
            "user_id": node,
            "degree": degree_dict.get(node, 0),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "degree_centrality": round(degree_centrality.get(node, 0.0), 4),
            "betweenness_centrality": round(betweenness_centrality.get(node, 0.0), 4),
            "community": community_map.get(node, "Cluster 1"),
            "direct_neighbors": neighbors,
            "aggressive_messages_sent": aggr_sent,
            "total_messages_sent": total_msgs
        }
        metrics_map[node] = node_info
        records.append({
            "User ID": node,
            "Degree": node_info["degree"],
            "Degree Centrality": node_info["degree_centrality"],
            "Betweenness Centrality": node_info["betweenness_centrality"],
            "Community": node_info["community"],
            "Aggressive Sent": node_info["aggressive_messages_sent"],
            "Neighbors": len(neighbors)
        })

    df_metrics = pd.DataFrame(records).sort_values(by="Betweenness Centrality", ascending=False)
    return metrics_map, df_metrics


def get_user_network_context(user_id: str, G: nx.Graph, metrics_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Retrieves focused network context for a specific user ID.
    Normalizes numeric IDs (e.g. 'usr_107', '107', 'usr_015').
    """
    if not user_id:
        return None

    target = user_id.strip()
    if target not in metrics_map:
        # Try zero-padded format: e.g., '15' -> 'usr_015'
        digits = "".join([ch for ch in target if ch.isdigit()])
        if digits:
            candidate = f"usr_{int(digits):03d}"
            if candidate in metrics_map:
                target = candidate
            else:
                candidate_short = f"usr_{int(digits)}"
                if candidate_short in metrics_map:
                    target = candidate_short

    if target not in metrics_map:
        return None

    node_data = metrics_map[target]
    return {
        "user_id": target,
        "is_known_in_graph": True,
        "degree": node_data["degree"],
        "degree_centrality": node_data["degree_centrality"],
        "betweenness_centrality": node_data["betweenness_centrality"],
        "community": node_data["community"],
        "neighbors": node_data["direct_neighbors"]
    }


# ==========================================
# VISUAL GRAPH 1: INTERACTIVE TOPOLOGY
# ==========================================
def create_network_figure(
    G: nx.Graph,
    metrics_map: Dict[str, Dict[str, Any]],
    highlight_user: Optional[str] = None
) -> go.Figure:
    """
    Renders an interactive force-directed Plotly network topology graph.
    """
    # Sample subgraph for layout clarity if node count is large
    display_nodes = list(G.nodes)[:60] if len(G.nodes) > 60 else list(G.nodes)
    if highlight_user and highlight_user in G and highlight_user not in display_nodes:
        display_nodes.append(highlight_user)

    subG = G.subgraph(display_nodes)
    pos = nx.spring_layout(subG, seed=42, k=0.35)

    edge_x = []
    edge_y = []
    hl_edge_x = []
    hl_edge_y = []

    for edge in subG.edges():
        u, v = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if highlight_user and (u == highlight_user or v == highlight_user):
            hl_edge_x.extend([x0, x1, None])
            hl_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    base_edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color="#cbd5e1"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    hl_edge_trace = go.Scatter(
        x=hl_edge_x, y=hl_edge_y,
        line=dict(width=2.5, color="#2563eb"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    palette = {
        "Cluster 1": "#2563eb",
        "Cluster 2": "#059669",
        "Cluster 3": "#7c3aed",
        "Cluster 4": "#d97706",
        "Cluster 5": "#0284c7"
    }

    node_x, node_y, node_color, node_size, node_text, hover_text = [], [], [], [], [], []

    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        info = metrics_map.get(node, {})
        comm = info.get("community", "Cluster 1")
        deg = info.get("degree", 1)
        bc = info.get("betweenness_centrality", 0.0)
        aggr = info.get("aggressive_messages_sent", 0)

        size = 11 + min(deg * 0.4, 18)
        color = palette.get(comm, "#64748b")

        if highlight_user and node == highlight_user:
            color = "#d97706"
            size += 10
        elif highlight_user and node in G.neighbors(highlight_user):
            color = "#0284c7"

        node_color.append(color)
        node_size.append(size)
        node_text.append(node)
        hover_text.append(
            f"<b>User:</b> {node}<br>"
            f"<b>Community:</b> {comm}<br>"
            f"<b>Degree:</b> {deg}<br>"
            f"<b>Betweenness Centrality:</b> {bc:.4f}<br>"
            f"<b>Aggressive Sent:</b> {aggr}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="#475569", family="Inter, sans-serif"),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="#ffffff")
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[base_edge_trace, hl_edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(b=15, l=15, r=15, t=15),
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=460
        )
    )
    return fig


# ==========================================
# VISUAL GRAPH 2: DEGREE DISTRIBUTION
# ==========================================
def create_degree_distribution_figure(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Renders the degree distribution histogram, illustrating scale-free social hub behavior.
    """
    fig = px.histogram(
        df_metrics,
        x="Degree",
        nbins=20,
        color_discrete_sequence=["#2563eb"],
        labels={"Degree": "Node Degree (Interaction Count)", "count": "Number of Users"}
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#475569", family="Inter, sans-serif"),
        margin=dict(t=30, b=25, l=35, r=20),
        height=280,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="User Count")
    )
    return fig


# ==========================================
# VISUAL GRAPH 3: TOP BETWEENNESS CENTRALITY
# ==========================================
def create_centrality_ranking_figure(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Renders Top 10 bridge nodes ranked by Betweenness Centrality.
    """
    top10 = df_metrics.head(10).sort_values(by="Betweenness Centrality", ascending=True)

    fig = go.Figure(go.Bar(
        x=top10["Betweenness Centrality"],
        y=top10["User ID"],
        orientation="h",
        marker=dict(color="#d97706"),
        text=[f"{bc:.4f}" for bc in top10["Betweenness Centrality"]],
        textposition="inside"
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#475569", family="Inter, sans-serif"),
        margin=dict(t=25, b=25, l=60, r=20),
        height=280,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Betweenness Centrality"),
        yaxis=dict(showgrid=False)
    )
    return fig


# ==========================================
# VISUAL GRAPH 4: PEERNESS VS AGGRESSION
# ==========================================
def create_peerness_aggression_figure(df_cb: pd.DataFrame) -> Optional[go.Figure]:
    """
    Visualizes Peerness vs Intent to Harm / Cyberbullying label from real dataset dyads.
    """
    if df_cb.empty or "Peerness" not in df_cb.columns:
        return None

    sample_df = df_cb.sample(n=min(len(df_cb), 600), random_state=42).copy()
    sample_df["Status"] = sample_df["CB_Label"].map({0: "Benign Communication", 1: "Cyberbullying Dyad"})

    fig = px.scatter(
        sample_df,
        x="Peerness",
        y="Intent_to_Harm",
        color="Status",
        color_discrete_map={"Benign Communication": "#059669", "Cyberbullying Dyad": "#dc2626"},
        opacity=0.65,
        labels={"Peerness": "Peerness (Social Distance)", "Intent_to_Harm": "Intent to Harm Score"}
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#475569", family="Inter, sans-serif"),
        margin=dict(t=25, b=25, l=35, r=20),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9")
    )
    return fig


# ==========================================
# VISUAL GRAPH 5: COMMUNITY CLUSTERS BREAKDOWN
# ==========================================
def create_community_breakdown_figure(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Visualizes the size and cumulative aggressive volume across detected communities.
    """
    comm_stats = df_metrics.groupby("Community").agg(
        User_Count=("User ID", "count"),
        Aggressive_Volume=("Aggressive Sent", "sum")
    ).reset_index()

    fig = go.Figure(data=[
        go.Bar(name="Member Users", x=comm_stats["Community"], y=comm_stats["User_Count"], marker_color="#2563eb"),
        go.Bar(name="Aggressive Sent", x=comm_stats["Community"], y=comm_stats["Aggressive_Volume"], marker_color="#dc2626")
    ])
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#475569", family="Inter, sans-serif"),
        margin=dict(t=25, b=25, l=35, r=20),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9")
    )
    return fig


# ==========================================
# VISUAL GRAPH 6: RECIPROCITY & CYBERBULLYING
# ==========================================
def create_reciprocity_figure(summary: Dict[str, Any]) -> go.Figure:
    """
    Donut chart of benign dyads vs cyberbullying dyads in the network.
    """
    labels = ["Benign Dyads", "Cyberbullying Dyads"]
    values = [summary.get("benign_dyads", 8519), summary.get("cb_dyads", 992)]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.6,
        color=labels,
        color_discrete_map={"Benign Dyads": "#059669", "Cyberbullying Dyads": "#dc2626"}
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#475569", family="Inter, sans-serif"),
        margin=dict(t=15, b=15, l=15, r=15),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=280
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig
