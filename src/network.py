"""
Social Network Analysis Module
------------------------------
This module handles network graph construction, centrality metric computations,
community detection, and Plotly graph generation.

INTEGRATION GUIDE FOR TEAMMATE:
To connect real platform interaction data:
1. Replace `load_interaction_graph()` with your data-loader function (e.g. from
   a database query, CSV edge list, or JSON interaction feed).
2. The graph should be a NetworkX Graph or DiGraph where nodes represent user IDs
   and edges represent interactions (replies, mentions, or quotes).
3. If edges have interaction weights or timestamps, add them as edge attributes:
   G.add_edge(u, v, weight=count).
"""

from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import pandas as pd


def load_synthetic_interaction_graph() -> nx.Graph:
    """
    Constructs a controlled synthetic interaction network for development and testing.
    CLEAR NOTICE: This network is synthetic and does not represent real platform users.
    
    Structure:
    - 20 synthetic user nodes (usr_101 through usr_120)
    - 3 distinct communities (cluster A, cluster B, bridge cluster C)
    - Key hub users with high degree and bridge users with high betweenness.
    """
    G = nx.Graph()

    # Community 1: Central moderation cluster (users 101 to 106)
    c1_edges = [
        ("usr_101", "usr_102"), ("usr_101", "usr_103"), ("usr_101", "usr_104"),
        ("usr_102", "usr_103"), ("usr_103", "usr_105"), ("usr_104", "usr_106"),
        ("usr_105", "usr_106"), ("usr_102", "usr_105")
    ]
    
    # Community 2: External high-volume group (users 107 to 113)
    c2_edges = [
        ("usr_107", "usr_108"), ("usr_107", "usr_109"), ("usr_108", "usr_109"),
        ("usr_109", "usr_110"), ("usr_110", "usr_111"), ("usr_111", "usr_112"),
        ("usr_112", "usr_113"), ("usr_108", "usr_113")
    ]

    # Community 3: Peripheral discussions group (users 114 to 120)
    c3_edges = [
        ("usr_114", "usr_115"), ("usr_115", "usr_116"), ("usr_116", "usr_117"),
        ("usr_117", "usr_118"), ("usr_118", "usr_119"), ("usr_119", "usr_120"),
        ("usr_114", "usr_120"), ("usr_116", "usr_119")
    ]

    # Inter-cluster bridges (critical for betweenness centrality)
    bridge_edges = [
        ("usr_103", "usr_107"),  # Bridge between Comm 1 and 2
        ("usr_106", "usr_114"),  # Bridge between Comm 1 and 3
        ("usr_110", "usr_117")   # Bridge between Comm 2 and 3
    ]

    for u, v in c1_edges + c2_edges + c3_edges + bridge_edges:
        G.add_edge(u, v, weight=1)

    return G


def compute_network_metrics(G: nx.Graph) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Calculates degree centrality, betweenness centrality, and detects communities.
    
    Returns:
        dict: Node-level metrics indexed by user_id
        DataFrame: Tabular metrics suitable for rendering in Streamlit.
    """
    if len(G.nodes) == 0:
        return {}, pd.DataFrame()

    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

    # Community detection using greedy modularity
    communities = list(nx.community.greedy_modularity_communities(G))
    community_map = {}
    for idx, comm in enumerate(communities, start=1):
        for node in comm:
            community_map[node] = f"Cluster {idx}"

    metrics_map: Dict[str, Dict[str, Any]] = {}
    records = []

    for node in G.nodes():
        node_info = {
            "user_id": node,
            "degree": degree_dict.get(node, 0),
            "degree_centrality": round(degree_centrality.get(node, 0.0), 3),
            "betweenness_centrality": round(betweenness_centrality.get(node, 0.0), 3),
            "community": community_map.get(node, "Unassigned"),
            "direct_neighbors": list(G.neighbors(node))
        }
        metrics_map[node] = node_info
        records.append({
            "User ID": node,
            "Degree": node_info["degree"],
            "Degree Centrality": node_info["degree_centrality"],
            "Betweenness Centrality": node_info["betweenness_centrality"],
            "Community": node_info["community"],
            "Neighbors Count": len(node_info["direct_neighbors"])
        })

    df_metrics = pd.DataFrame(records).sort_values(by="Betweenness Centrality", ascending=False)
    return metrics_map, df_metrics


def get_user_network_context(user_id: str, G: nx.Graph, metrics_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Retrieves focused network context for a specific user ID.
    Used by the recommendation engine and user inspection UI.
    """
    if not user_id or user_id not in G:
        return None

    node_data = metrics_map.get(user_id)
    if not node_data:
        return None

    neighbors = list(G.neighbors(user_id))
    return {
        "user_id": user_id,
        "is_known_in_graph": True,
        "degree": len(neighbors),
        "degree_centrality": node_data["degree_centrality"],
        "betweenness_centrality": node_data["betweenness_centrality"],
        "community": node_data["community"],
        "neighbors": neighbors
    }


def create_network_figure(
    G: nx.Graph,
    metrics_map: Dict[str, Dict[str, Any]],
    highlight_user: Optional[str] = None
) -> go.Figure:
    """
    Builds an interactive Plotly graph visualization adhering to the dark slate palette.
    """
    # Deterministic layout for consistent rendering
    pos = nx.spring_layout(G, seed=42, k=0.45)

    # Edge traces
    edge_x = []
    edge_y = []
    highlight_edge_x = []
    highlight_edge_y = []

    for edge in G.edges():
        u, v = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        if highlight_user and (u == highlight_user or v == highlight_user):
            highlight_edge_x.extend([x0, x1, None])
            highlight_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    base_edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.0, color="#cbd5e1"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    highlight_edge_trace = go.Scatter(
        x=highlight_edge_x, y=highlight_edge_y,
        line=dict(width=2.4, color="#2563eb"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

    # Community color scheme (Clean, accessible light theme palette)
    palette = {
        "Cluster 1": "#2563eb",  # Blue
        "Cluster 2": "#059669",  # Emerald
        "Cluster 3": "#7c3aed",  # Purple
        "Unassigned": "#64748b"
    }

    # Node traces
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    hover_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        info = metrics_map.get(node, {})
        comm = info.get("community", "Unassigned")
        deg = info.get("degree", 1)
        bc = info.get("betweenness_centrality", 0.0)

        # Base node sizing with cap
        size = 13 + (deg * 2.5)
        color = palette.get(comm, "#64748b")

        if highlight_user and node == highlight_user:
            color = "#d97706"  # Amber highlight
            size = size + 8
        elif highlight_user and node in G.neighbors(highlight_user):
            color = "#0284c7"  # Sky blue for direct neighbors

        node_color.append(color)
        node_size.append(size)
        node_text.append(node)
        
        hover_info = (
            f"<b>User:</b> {node}<br>"
            f"<b>Community:</b> {comm}<br>"
            f"<b>Degree (Connections):</b> {deg}<br>"
            f"<b>Betweenness Centrality:</b> {bc:.3f}"
        )
        hover_text.append(hover_info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color="#475569", family="Inter, sans-serif"),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2.0, color="#ffffff")
        ),
        showlegend=False
    )

    fig = go.Figure(
        data=[base_edge_trace, highlight_edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(b=20, l=20, r=20, t=30),
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=480
        )
    )

    return fig
