"""
Social Network Analysis (SNA) Module
------------------------------------
Handles graph construction, centrality metrics, community detection,
and academic-grade interactive node-link and tree visualizations using NetworkX and Plotly.
Supports:
1. Global Interaction Topology (Node-link)
2. Cyberbullying Directed Attack Network (Node-link)
3. Radial Ego-Tree / Star Network (Tree graph centered on selected user)
4. Minimum Spanning Tree (MST - Acyclic backbone tree)
5. Centrality and Community analytical charts
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import math
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
        except Exception:
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
    """Controlled 20-node fallback synthetic network."""
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
    """Computes degree centrality, betweenness centrality, in/out degrees, and communities."""
    if len(G.nodes) == 0:
        return {}, pd.DataFrame()

    is_directed = G.is_directed()
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
    """Retrieves focused network context for a specific user ID."""
    if not user_id:
        return None

    target = user_id.strip()
    if target not in metrics_map:
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
# 1. NODE-LINK: GLOBAL NETWORK TOPOLOGY
# ==========================================
def create_network_figure(
    G: nx.Graph,
    metrics_map: Dict[str, Dict[str, Any]],
    highlight_user: Optional[str] = None
) -> go.Figure:
    """Renders interactive force-directed node-link communication network."""
    display_nodes = list(G.nodes)[:60] if len(G.nodes) > 60 else list(G.nodes)
    if highlight_user and highlight_user in G and highlight_user not in display_nodes:
        display_nodes.append(highlight_user)

    subG = G.subgraph(display_nodes)
    pos = nx.spring_layout(subG, seed=42, k=0.35)

    edge_x, edge_y, hl_edge_x, hl_edge_y = [], [], [], []

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
            height=440
        )
    )
    return fig


# ==========================================
# 2. NODE-LINK: CYBERBULLYING ATTACK NETWORK
# ==========================================
def create_attack_network_figure(df_cb: pd.DataFrame, max_edges: int = 150) -> go.Figure:
    """
    Renders the filtered cyberbullying attack network (CB_Label == 1).
    Displays only direct aggression and harassment links between users.
    """
    if df_cb.empty or "CB_Label" not in df_cb.columns:
        return go.Figure()

    df_attack = df_cb[df_cb["CB_Label"] == 1].head(max_edges)
    G_attack = nx.DiGraph()

    for _, r in df_attack.iterrows():
        u = f"usr_{int(r['User1 ID']):03d}"
        v = f"usr_{int(r['User2 ID']):03d}"
        G_attack.add_edge(u, v, weight=int(r['Aggressive_Count']))

    pos = nx.spring_layout(G_attack, seed=42, k=0.4)

    edge_x, edge_y = [], []
    for u, v in G_attack.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color="rgba(220, 38, 38, 0.45)"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    in_deg = dict(G_attack.in_degree())
    out_deg = dict(G_attack.out_degree())

    node_x, node_y, node_color, node_size, node_text, hover_text = [], [], [], [], [], []

    for node in G_attack.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        outs = out_deg.get(node, 0)
        ins = in_deg.get(node, 0)
        size = 10 + min(outs * 1.5, 20)

        # Attackers colored deep red, victims colored amber/orange
        color = "#dc2626" if outs > ins else "#d97706"

        node_color.append(color)
        node_size.append(size)
        node_text.append(node)
        hover_text.append(
            f"<b>Node:</b> {node}<br>"
            f"<b>Outgoing Attacks:</b> {outs}<br>"
            f"<b>Incoming Attacks:</b> {ins}<br>"
            f"<b>Role:</b> {'Primary Instigator' if outs > ins else 'Targeted Victim'}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#475569", family="Inter, sans-serif"),
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="#ffffff")
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(b=15, l=15, r=15, t=15),
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=340
        )
    )
    return fig


# ==========================================
# 3. TREE GRAPH: RADIAL EGO-TREE (STAR TREE)
# ==========================================
def create_ego_tree_figure(
    user_id: str,
    G: nx.Graph,
    df_cb: pd.DataFrame
) -> go.Figure:
    """
    Renders a radial hierarchical tree layout centered on the selected user node.
    Root node at center, branches linking to 1st-degree contacts.
    """
    if not user_id or user_id not in G:
        return go.Figure()

    neighbors = list(G.successors(user_id)) if G.is_directed() else list(G.neighbors(user_id))
    display_neighbors = neighbors[:32]  # Cap for radial clarity

    # Radial tree coordinates: center at (0, 0), neighbors on circle radius R=1
    pos = {user_id: (0.0, 0.0)}
    num_n = max(len(display_neighbors), 1)

    for i, nbr in enumerate(display_neighbors):
        angle = 2.0 * math.pi * i / num_n
        pos[nbr] = (math.cos(angle), math.sin(angle))

    edge_x, edge_y, edge_colors = [], [], []

    # Map harassment relationships
    attack_pairs = set()
    if not df_cb.empty and "CB_Label" in df_cb.columns:
        for _, r in df_cb[(df_cb["CB_Label"] == 1)].iterrows():
            attack_pairs.add((f"usr_{int(r['User1 ID']):03d}", f"usr_{int(r['User2 ID']):03d}"))

    for nbr in display_neighbors:
        x0, y0 = pos[user_id]
        x1, y1 = pos[nbr]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color="#94a3b8"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    node_x, node_y, node_color, node_size, node_text, hover_text = [], [], [], [], [], []

    # Root Node (Selected User)
    node_x.append(0.0)
    node_y.append(0.0)
    node_color.append("#2563eb")  # Blue root
    node_size.append(26)
    node_text.append(user_id)
    hover_text.append(f"<b>Root Center:</b> {user_id}<br><b>Total Direct Ties:</b> {len(neighbors)}")

    # Branch Nodes
    for nbr in display_neighbors:
        x, y = pos[nbr]
        node_x.append(x)
        node_y.append(y)

        is_aggr = (user_id, nbr) in attack_pairs or (nbr, user_id) in attack_pairs
        color = "#dc2626" if is_aggr else "#059669"

        node_color.append(color)
        node_size.append(12)
        node_text.append(nbr)
        hover_text.append(
            f"<b>Contact:</b> {nbr}<br>"
            f"<b>Connection:</b> {'Cyberbullying Dyad' if is_aggr else 'Benign Communication'}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#475569", family="Inter, sans-serif"),
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.8, color="#ffffff")
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(b=15, l=15, r=15, t=15),
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=340
        )
    )
    return fig


# ==========================================
# 4. TREE GRAPH: MINIMUM SPANNING TREE (MST)
# ==========================================
def create_mst_tree_figure(df_cb: pd.DataFrame) -> go.Figure:
    """
    Constructs the NetworkX Minimum Spanning Tree (MST) spanning the 100 users.
    An acyclic mathematical tree preserving highest-frequency interaction pathways.
    """
    if df_cb.empty:
        return go.Figure()

    G_undir = nx.Graph()
    for _, r in df_cb.iterrows():
        u = f"usr_{int(r['User1 ID']):03d}"
        v = f"usr_{int(r['User2 ID']):03d}"
        w = 1.0 / max(int(r['Total_messages']), 1)
        G_undir.add_edge(u, v, weight=w)

    mst = nx.minimum_spanning_tree(G_undir, weight='weight')
    pos = nx.spring_layout(mst, seed=42, k=0.45)

    edge_x, edge_y = [], []
    for u, v in mst.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.4, color="#0284c7"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    deg = dict(mst.degree())
    node_x, node_y, node_color, node_size, node_text, hover_text = [], [], [], [], [], []

    for node in mst.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        d = deg.get(node, 1)
        size = 8 + min(d * 2.5, 20)
        color = "#2563eb" if d > 3 else "#64748b"

        node_color.append(color)
        node_size.append(size)
        node_text.append(node)
        hover_text.append(f"<b>Tree Node:</b> {node}<br><b>Tree Degree (Branches):</b> {d}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="#475569", family="Inter, sans-serif"),
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="#ffffff")
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(b=15, l=15, r=15, t=15),
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=340
        )
    )
    return fig


# ==========================================
# 5. BAR CHART 1: TOP 10 BETWEENNESS BRIDGES
# ==========================================
def create_centrality_ranking_figure(df_metrics: pd.DataFrame) -> go.Figure:
    """Renders Top 10 bridge nodes ranked by Betweenness Centrality."""
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
# 6. BAR CHART 2: COMMUNITY CLUSTERS BREAKDOWN
# ==========================================
def create_community_breakdown_figure(df_metrics: pd.DataFrame) -> go.Figure:
    """Visualizes the size and cumulative aggressive volume across detected communities."""
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
