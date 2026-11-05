# CyberShield AI

A human-reviewed cyberbullying detection and social network analysis (SNA) moderation platform designed for trust and safety operations.

CyberShield AI pairs text-based toxicity classification with social graph structural context (degree centrality, betweenness centrality, community clusters) to help human moderators prioritize and evaluate harmful content without automated punitive actions (bans or deletions).

---

## Architecture Overview

The codebase is organized into cleanly separated modules so that frontend UI development and backend machine learning / graph analytics can be developed in parallel:

```
SNA  PROJECT/
├── .streamlit/
│   └── config.toml             # Streamlit light theme & runtime settings
├── src/
│   ├── __init__.py
│   ├── classifier.py           # NLP classifier integration & mock baseline
│   ├── network.py              # NetworkX analysis, centrality, & Plotly renderer
│   ├── recommender.py          # Rule-based decision logic (Classifier + Network)
│   ├── data_store.py           # Sample audit stream & queue validation
│   └── styles.py               # Clean modern light CSS design system
├── app.py                      # Main Streamlit dashboard application
├── test_integration.py         # Pipeline verification test suite
├── requirements.txt            # Project dependencies
└── README.md                   # System documentation & integration manual
```

### Module Responsibilities

1. **`app.py`**: Handles page navigation, dashboard layouts, metrics rendering, interactive user inputs, and session state persistence across the 4 core views:
   - **Overview**: Audit metrics, classification distribution donut chart, multi-agent status, and recent moderation logs.
   - **Comment Analysis**: Single comment text submission, batch CSV/JSON file upload, mock prediction breakdown, and queue dispatch.
   - **Network Analysis**: 4 Node-Link & Tree network visualizers, 2 quantitative bar charts, user inspection drawer, and betweenness centrality ledger.
   - **Review Queue**: Moderator review workbench with approval, confirmation, and dismissal actions.
2. **`src/classifier.py`**: Defines the `ClassifierResult` data contract and the mock heuristic engine. **This is where the trained ML model is integrated.**
3. **`src/network.py`**: Connects the 100-user Mendeley dataset (90,356 messages, 9,511 dyads), computes NetworkX metrics, and builds interactive Plotly node-link and tree visualizations (Attack Network, Radial Ego-Tree, MST Backbone Tree).
4. **`src/recommender.py`**: Evaluates classifier output and network context under an ethical rule system (`Allow`, `Review`, `Flag`). Network metrics indicate reach and amplification potential—they are **never** treated as proof of guilt.
5. **`src/data_store.py`**: Manages sample moderation stream records, batch file processing, review queue operations, and input validation.
6. **`src/styles.py`**: Clean, accessible light SaaS design system inspired by Stripe and GitHub.

---

## Multi-Agent AI Architecture (Agentic AI)

CyberShield AI implements a **4-Agent Collaborative Moderation Pipeline**:

1. **Agent 1: NLP Content Classifier Agent (Agentic NLP)**
   - Scans incoming comment text for toxicity, aggressive intent, and harassment patterns.
   - Produces normalized confidence scores and class probability distributions.
2. **Agent 2: Social Network Analyst Agent (Agentic SNA via NetworkX)**
   - Computes structural network properties: betweenness centrality, degree distribution, community modularity, and communication reciprocity.
   - Identifies bridge users who act as bottlenecks for viral amplification across clusters.
3. **Agent 3: Decision & Triage Agent (Rule Orchestrator)**
   - Synthesizes text classification and topological reach features to determine action recommendations (`Allow`, `Review`, `Flag`) with urgency tiers.
   - Generates transparent, human-readable audit explanations.
4. **Agent 4: Human-in-the-Loop Moderator Agent (Supervisory Controller)**
   - Provides final human verification in the Review Queue without automated account penalties or bans.

---

## Social Network Analysis (SNA) Visual Suite

The platform provides an extensive suite of interactive node-link and tree visualizations:

* **Graph 1: Global Interaction Topology (Node-Link Network)** — Full communication graph with community coloring and neighborhood highlighting.
* **Graph 2: Cyberbullying Attack Network (Directed Node-Link)** — Filtered view isolating the 992 cyberbullying relationships, identifying instigators (red) and targets (amber).
* **Graph 3: Radial Ego-Tree (Hierarchical Star Tree)** — Tree layout centered on a selected root user, branching out to their direct contacts color-coded by dyad safety.
* **Graph 4: Minimum Spanning Tree (MST Backbone Tree)** — NetworkX acyclic tree (`nx.minimum_spanning_tree`) connecting all 100 users through the strongest communication paths without redundant loops.
* **Graph 5: Top 10 Betweenness Centrality Bridges (Bar Chart)** — Identifies the critical bridge accounts connecting disparate student clusters.
* **Graph 6: Community Clusters & Aggressive Volume (Grouped Bar Chart)** — Compares user count vs. cumulative aggressive message volume across detected communities.

---

## Getting Started

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` or `uv`

### Installation

1. **Clone or open the workspace:**
   ```bash
   cd "SNA  PROJECT"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify module integration:**
   ```bash
   python test_integration.py
   ```

5. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

Open your browser at `http://localhost:8501`.

---

## Current Mock Components

To support frontend design and verification before backend modeling is complete, the platform currently runs in **Demonstration Mode**:

| Component | Current State | File Location |
| :--- | :--- | :--- |
| **Comment Classifier** | Rule-based mock keyword heuristic (`MockClassifier`) returning labels (`Clean`, `Toxic`, `Harassment`) and confidence scores. | `src/classifier.py` |
| **Interaction Graph** | Controlled 20-node synthetic interaction network with 3 distinct communities, hubs, and bridges. | `src/network.py` |
| **Audit Stream / Queue** | Initial sample records representing typical trust & safety audit comments. | `src/data_store.py` |

All mock components are clearly labeled with badges in the UI so that sample inferences are never confused with production AI predictions.

---

## Teammate Integration Guide

### 1. Integrating the Machine Learning Classifier (`src/classifier.py`)

Open `src/classifier.py` and replace `MockClassifier.predict()` or the `predict_comment()` function:

1. Load your model weights and tokenizer/vectorizer:
   ```python
   # Example: HuggingFace pipeline or PyTorch/scikit-learn model
   from transformers import pipeline

   model_pipeline = pipeline("text-classification", model="path/to/trained_model")
   ```
2. Map your model's prediction into `ClassifierResult`:
   ```python
   return ClassifierResult(
       predicted_label="Toxic",          # or "Harassment", "Clean"
       confidence=0.92,                  # float between 0.0 and 1.0
       class_scores={"Toxic": 0.92, "Clean": 0.05, "Harassment": 0.03},
       is_mock=False,                    # Set to False once genuine model is loaded
       model_version="bert-cyberbullying-v1.0",
       inference_notes="Inference executed via fine-tuned transformer."
   )
   ```
3. Because the UI consumes `ClassifierResult`, **no changes to `app.py` are needed.**

### 2. Integrating Real Interaction Graph Data (`src/network.py`)

Open `src/network.py` and replace `load_synthetic_interaction_graph()`:

1. Replace synthetic edge construction with your real graph data loader (e.g., CSV edge list, database query, or JSON platform export):
   ```python
   def load_interaction_graph() -> nx.Graph:
       # Example: Loading from an edge-list CSV
       df_edges = pd.read_csv("data/user_interactions.csv")
       G = nx.from_pandas_edgelist(
           df_edges, 
           source="source_user_id", 
           target="target_user_id", 
           edge_attr=["weight"]
       )
       return G
   ```
2. The existing functions `compute_network_metrics()`, `get_user_network_context()`, and `create_network_figure()` automatically compute centrality, detect communities, and generate the Plotly visualization.
3. No changes to `app.py` are required.

---

## Moderation Principles & Safety Guardrails

- **Human In The Loop**: CyberShield AI does not automatically delete comments, block user accounts, or trigger punitive consequences.
- **Contextual Network Metrics**: High betweenness centrality or high degree indicates potential broadcast reach and audience exposure; it does not indicate malicious intent or culpability.
- **Audit Trails**: Every action taken by a moderator in the Review Queue updates the internal audit state.
