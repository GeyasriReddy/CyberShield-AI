import streamlit
import plotly
import networkx as nx
import pandas as pd
from src.classifier import predict_comment
from src.network import (
    load_dataset_interaction_graph,
    compute_network_metrics,
    get_user_network_context,
    create_network_figure,
    create_attack_network_figure,
    create_ego_tree_figure,
    create_mst_tree_figure,
    create_centrality_ranking_figure,
    create_community_breakdown_figure
)
from src.recommender import generate_recommendation
from src.data_store import get_initial_sample_comments, calculate_overview_metrics

print("Imports successful")
clf = predict_comment("You are an idiot")
print("Classifier output:", clf.predicted_label, clf.confidence)

G, df_cb, summary = load_dataset_interaction_graph()
metrics_map, df_metrics = compute_network_metrics(G)
print(f"Graph nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
print("Graph source:", summary.get("source"))
print(f"Centrality records: {len(df_metrics)}")

rec = generate_recommendation(clf, metrics_map.get("usr_015"))
print(f"Recommendation: {rec.action} | Reason: {rec.primary_reason}")

comments = get_initial_sample_comments()
overview = calculate_overview_metrics(comments)
print("Overview metrics:", overview)

fig1 = create_network_figure(G, metrics_map, highlight_user="usr_015")
fig2 = create_attack_network_figure(df_cb, max_edges=60)
fig3 = create_ego_tree_figure("usr_015", G, df_cb)
fig4 = create_mst_tree_figure(df_cb)
fig5 = create_centrality_ranking_figure(df_metrics)
fig6 = create_community_breakdown_figure(df_metrics)

print("Figure 1 (Topology) traces:", len(fig1.data))
print("Figure 2 (Attack Network) traces:", len(fig2.data))
print("Figure 3 (Radial Ego-Tree) traces:", len(fig3.data))
print("Figure 4 (MST Tree) traces:", len(fig4.data))
print("Figure 5 (Centrality Bar) traces:", len(fig5.data))
print("Figure 6 (Community Bar) traces:", len(fig6.data))
print("ALL NODE & TREE VISUALIZATION CHECKS PASSED!")
