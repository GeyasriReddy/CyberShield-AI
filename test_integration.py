import streamlit
import plotly
import networkx as nx
import pandas as pd
from src.classifier import predict_comment
from src.network import (
    load_synthetic_interaction_graph,
    compute_network_metrics,
    get_user_network_context,
    create_network_figure
)
from src.recommender import generate_recommendation
from src.data_store import get_initial_sample_comments, calculate_overview_metrics

print("Imports successful")
clf = predict_comment("You are an idiot")
print("Classifier output:", clf.predicted_label, clf.confidence)

G = load_synthetic_interaction_graph()
metrics_map, df_metrics = compute_network_metrics(G)
print(f"Graph nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
print(f"Centrality records: {len(df_metrics)}")

rec = generate_recommendation(clf, metrics_map.get("usr_107"))
print(f"Recommendation: {rec.action} | Reason: {rec.primary_reason}")

comments = get_initial_sample_comments()
overview = calculate_overview_metrics(comments)
print("Overview metrics:", overview)

fig = create_network_figure(G, metrics_map, highlight_user="usr_103")
print("Figure traces:", len(fig.data))
print("ALL INTEGRATION CHECKS PASSED!")
