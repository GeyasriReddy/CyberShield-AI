# CyberShield AI — Phase 5: Social Network Analysis (SNA) Implementation Report
**Document ID:** `docs/SNA_IMPLEMENTATION_REPORT.md`  
**Project:** CyberShield AI — Final-Year B.Tech Project  
**Author / Responsibility:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** November 5, 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Phase 5 transitions the CyberShield AI project from isolated natural language processing to graph-based relational modeling. By synthesizing the calibrated NLP aggression model (`models/best_model_pipeline.joblib`) with longitudinal peer-to-peer communication records, this phase constructs dual directed network representations and extracts comprehensive node- and dyad-level Social Network Analysis (SNA) metrics.

All implementations strictly adhere to structural integrity and data leakage prevention mandates:
1. **Zero Raw Dataset Modifications:** Original CSVs in `datasets/` were preserved intact.
2. **Zero Upstream Pipeline Modification:** Phase 3 text classification models and datasets were utilized in read-only inference mode.
3. **Absolute Target Leakage Prevention:** Neither `CB_Label`, `Aggressive_Count`, nor `Intent_to_Harm` from `cb_labels_cleaned.csv` were accessed or incorporated into any feature matrix.
4. **Peerness Segregation:** The deterministic demographic peerness score is isolated as `exploratory_peerness_score` solely for descriptive and exploratory analysis, preventing structural target contamination during subsequent Stage 2 modeling.
5. **No Stage 2 Modeling:** Model training for dyadic cyberbullying detection is explicitly deferred to Phase 6.

All 46 unit tests in the CyberShield AI test suite passed cleanly.

---

## 2. Graph Architectures and Schemas

Two distinct directed NetworkX graph representations were constructed and exported in standard GraphML format to support multi-scale relational analytics:

```
+---------------------------------------------------------------------------------------------------+
|                                      CyberShield AI Graph Suite                                   |
+---------------------------------------------------------------------------------------------------+
|  Graph A: Temporal MultiDiGraph                              Graph B: Aggregated DiGraph          |
|  File: results/communication_multidigraph.graphml            File: results/communication_digraph.graphml |
|                                                                                                   |
|  - Nodes: 100 registered students                            - Nodes: 100 registered students     |
|  - Edges: 90,356 directed message events                     - Edges: 9,511 directed dyads       |
|  - Self-loops: Exactly 0                                     - Self-loops: Exactly 0             |
|  - Type: Dynamic, event-level multigraph                     - Type: Static weighted graph        |
|  - Edge Data: timestamp, text, prob, pred                    - Edge Data: count, prob stats, span|
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Graph A: Temporal Communication MultiDiGraph
- **Target File:** `results/communication_multidigraph.graphml`
- **Graph Class:** `networkx.MultiDiGraph`
- **Total Nodes:** 100 (User IDs 1 through 100)
- **Total Directed Edges:** 90,356 (one edge per individual communication event)
- **Self-Loops:** Strictly 0 (no student messaged themselves)
- **Node Attributes:**
  - `user_id` (`int`): Registered User ID (1 to 100)
  - `age` (`int`): Student age
  - `gender` (`str`): "Boy" / "Girl"
  - `school` (`str`): Anonymized school identifier
  - `grade` (`int`): Student academic grade level
- **Edge Attributes:**
  - `message_id` (`int`): Sequential index from sorted communication corpus
  - `timestamp` (`str`): ISO-formatted chronological timestamp (`YYYY-MM-DD HH:MM:SS`)
  - `message_text` (`str`): Original communication text
  - `aggression_probability` (`float`): Calibrated model probability $P(\text{Aggressive} \mid \text{text}) \in [0.0, 1.0]$
  - `aggression_prediction` (`int`): Binary classification ($1$ if $P \ge 0.5$, else $0$)
  - `prediction_status` (`str`): Audit status tag (`"inferred"`)

### 2.2 Graph B: Aggregated Interaction DiGraph
- **Target File:** `results/communication_digraph.graphml`
- **Graph Class:** `networkx.DiGraph`
- **Total Nodes:** 100 (User IDs 1 through 100)
- **Total Directed Edges:** 9,511 (unique ordered communicating pairs $u \to v$)
- **Self-Loops:** Strictly 0
- **Network Density:** $0.9607$ ($9,511 / 9,900$ possible directed dyads)
- **Edge Attributes:**
  - `total_messages` (`int`): Total volume of messages sent from $u \to v$
  - `predicted_aggression_volume` (`float`): Cumulative aggression probability mass $\sum P$
  - `mean_aggression_prob` (`float`): Mean aggression probability $\frac{1}{N}\sum P$
  - `max_aggression_prob` (`float`): Maximum observed single-event probability $\max(P)$
  - `interaction_span_days` (`float`): Chronological span between first and last message in days
  - `distance` (`float`): Topological routing distance metric, defined as $\frac{1}{N_{\text{msgs}} + 1}$ for shortest-path algorithms

---

## 3. Integration of NLP Model Predictions

The Stage 1 text classification pipeline (`models/best_model_pipeline.joblib`), trained and validated in Phase 3, was applied across all 90,356 cleaned communication events.

### 3.1 Inference Characteristics
- **Total Inferred Messages:** 90,356 records
- **Probability Range:** Minimum $= 0.000005$, Maximum $= 1.000000$
- **Mean Aggression Probability:** $0.3174$
- **Median Aggression Probability:** $0.1843$
- **Aggression Classification ($P \ge 0.5$):**
  - Predicted Non-Aggressive ($0$): 64,174 messages ($71.02\%$)
  - Predicted Aggressive ($1$): 26,182 messages ($28.98\%$)
- **Missing Value Handling:** In Phase 2 data remediation, null and unparseable communication records were systematically resolved. Zero NaN or unassigned values were present during graph mapping.
- **Output Artifact:** `data/processed/communication_with_predictions.csv` (90,356 rows $\times$ 6 columns)

---

## 4. Engineered Social Network Analysis Features

Feature extraction was performed at two distinct analytical granularities: individual node positions and pairwise dyadic interactions.

### 4.1 Node-Level Metrics (`results/node_sna_features.csv`)
Comprises 100 rows (one per user) and 22 feature attributes:

| Feature Name | Type | Definition & Mathematical Rationale |
| :--- | :--- | :--- |
| `user_id` | Integer | Unique identifier for registered student ($1 \dots 100$) |
| `age`, `gender`, `school`, `grade` | Demographic | Basic student profile metadata |
| `in_degree` | Integer | Total unique peers who sent at least one message to user |
| `out_degree` | Integer | Total unique peers to whom user sent at least one message |
| `in_strength` | Integer | Total incoming message volume across all peers: $\sum_{v} w_{v \to u}$ |
| `out_strength` | Integer | Total outgoing message volume across all peers: $\sum_{v} w_{u \to v}$ |
| `total_activity_strength` | Integer | Cumulative communication involvement: $\text{in\_strength} + \text{out\_strength}$ |
| `activity_imbalance` | Float | Directional skew in communication volume: $\frac{\text{out} - \text{in}}{\text{total} + \epsilon} \in [-1.0, 1.0]$ |
| `incoming_aggression_volume` | Float | Total predicted aggression probability mass received: $\sum_{v} \sum_{m} P_{v \to u}(m)$ |
| `outgoing_aggression_volume` | Float | Total predicted aggression probability mass sent: $\sum_{v} \sum_{m} P_{u \to v}(m)$ |
| `mean_incoming_aggression_prob` | Float | Mean toxicity of messages received: $\frac{\text{incoming\_aggression\_volume}}{\text{in\_strength}}$ |
| `mean_outgoing_aggression_prob` | Float | Mean toxicity of messages transmitted: $\frac{\text{outgoing\_aggression\_volume}}{\text{out\_strength}}$ |
| `aggression_transmission_ratio` | Float | Aggression output vs. input ratio: $\frac{\text{outgoing\_volume}}{\text{incoming\_volume} + 1.0}$ |
| `pagerank_message_weighted` | Float | Stationary random walk probability weighted by raw message traffic |
| `pagerank_aggression_weighted` | Float | Stationary random walk probability weighted by predicted aggression volume |
| `betweenness_centrality` | Float | Fraction of all shortest communication paths passing through user (using $d = \frac{1}{w+1}$) |
| `hub_score` | Float | Kleinberg HITS Hub score reflecting outgoing connection prominence |
| `authority_score` | Float | Kleinberg HITS Authority score reflecting incoming endorsement prominence |
| `clustering_coefficient` | Float | Local transitivity / density of user's immediate neighborhood |

### 4.2 Dyad-Level Metrics (`results/dyad_sna_features.csv`)
Comprises 9,511 rows (one per directed communicating dyad $u_1 \to u_2$) and 21 feature attributes:

| Feature Name | Type | Definition & Mathematical Rationale |
| :--- | :--- | :--- |
| `user1_id` | Integer | Sender user ID |
| `user2_id` | Integer | Receiver user ID |
| `total_messages` | Integer | Direct message count transmitted from $u_1 \to u_2$ ($N_{\text{forward}}$) |
| `mean_aggression_probability` | Float | Average predicted aggression probability for $u_1 \to u_2$ messages |
| `predicted_aggression_volume` | Float | Cumulative aggression probability sum: $\sum_{m \in (u_1 \to u_2)} P(m)$ |
| `max_aggression_probability` | Float | Peak single-event aggression score observed from $u_1 \to u_2$ |
| `reciprocal_messages` | Integer | Reverse message count transmitted from $u_2 \to u_1$ ($N_{\text{reverse}}$) |
| `reciprocity_ratio` | Float | Reverse response share: $\frac{N_{\text{reverse}}}{N_{\text{forward}} + N_{\text{reverse}}}$ |
| `directional_communication_imbalance` | Float | Dyadic transmission asymmetry: $\frac{N_{\text{forward}} - N_{\text{reverse}}}{N_{\text{forward}} + N_{\text{reverse}} + \epsilon}$ |
| `interaction_span_days` | Float | Calendar duration (days) from first to last message between $u_1$ and $u_2$ |
| `inter_message_time_mean_hours` | Float | Mean inter-arrival time (hours) between consecutive $u_1 \to u_2$ messages |
| `inter_message_time_std_hours` | Float | Standard deviation of inter-arrival times (measures temporal burstiness) |
| `sender_out_strength` | Integer | Global outgoing volume of sender $u_1$ |
| `receiver_in_strength` | Integer | Global incoming volume of receiver $u_2$ |
| `sender_outgoing_aggression_volume` | Float | Global aggression transmitted across all recipients by sender $u_1$ |
| `receiver_incoming_aggression_exposure`| Float | Global aggression received from all peers by receiver $u_2$ |
| `sender_pagerank` | Float | Message-weighted PageRank of sender $u_1$ |
| `receiver_pagerank` | Float | Message-weighted PageRank of receiver $u_2$ |
| `sender_hub_score` | Float | HITS Hub score of sender $u_1$ |
| `receiver_authority_score` | Float | HITS Authority score of receiver $u_2$ |
| `exploratory_peerness_score` | Float | Recovered demographic peerness score (retained strictly for exploratory analysis) |

---

## 5. Structural Network Insights and Findings

Analysis of `results/sna_summary.csv` and the topological distributions reveals critical properties that dictate Stage 2 modeling strategy:

### 5.1 Extreme Network Density ($96.07\%$)
- In a cohort of 100 students, there exist $100 \times 99 = 9,900$ possible directed interactions.
- 9,511 of these pairs exchanged at least one message, yielding a network density of $0.9607$.
- **Architectural Implication:** Unweighted topological metrics (e.g., standard degree centrality, unweighted shortest paths) have near-zero variance and cannot effectively distinguish cyberbullying dynamics.
- **Solution:** Discriminating power is concentrated in **weighted interaction intensity**, **cumulative aggression probability volume**, **temporal burstiness**, and **severe directional reciprocity imbalance**.

### 5.2 Perfect Asymmetry in Cyberbullying Dyads
- Out of all 9,900 dyads, exactly 992 are labeled as positive cyberbullying cases in the dataset ground truth.
- **Mutual Cyberbullying Analysis:** Across all 992 positive dyads, the occurrence of mutual cyberbullying (where both $u_1 \to u_2 = 1$ and $u_2 \to u_1 = 1$) is **strictly 0.00%**.
- In every recorded incident of cyberbullying, the relationship is strictly asymmetric. This confirms that cyberbullying represents a systematic power imbalance rather than bidirectional mutual conflict, justifying the directed graph topology.

### 5.3 Communication Reciprocity and Asymmetry
- $9,144$ dyads ($96.14\%$) exhibit bidirectional communication ($N_{\text{reverse}} > 0$).
- $367$ dyads ($3.86\%$) exhibit strictly one-way communication ($N_{\text{reverse}} = 0$).
- Mean messages per directed dyad is $9.50$.
- The directional communication imbalance metric spans the full spectrum $[-1.0, 1.0]$, capturing severe conversational dominance.

### 5.4 Aggression Extremes
- **Highest Aggression Transmitter:** User 1 produced the greatest cumulative outgoing aggression volume ($437.19$ probability mass across peers).
- **Highest Aggression Target:** User 39 received the greatest cumulative incoming aggression exposure ($356.42$ probability mass across peers).

---

## 6. Rigorous Target Leakage Safeguards

To maintain scientific integrity and prevent invalid predictive claims, strict safeguards were enforced throughout feature engineering:

### 6.1 Exclusion of Heuristic Labeling Formula Components
During Phase 4 data inspection, the ground truth generation process of the Mendeley dataset was reverse-engineered:
A shallow decision tree trained on `Aggressive_Count`, `Intent_to_Harm`, and `Peerness` reconstructed the ground truth `CB_Label` with **$98.03\%$ accuracy**.

This proves that `Aggressive_Count` and `Intent_to_Harm` are not independent empirical observations, but the dataset authors' internal labeling heuristic. Consequently:
- `CB_Label` is completely omitted from all feature matrices.
- `Aggressive_Count` and `Intent_to_Harm` are strictly excluded from all feature matrices.
- The CyberShield AI Stage 2 model will predict cyberbullying strictly using **raw communication dynamics**, **unsupervised network centralities**, and **independent Stage 1 NLP probability distributions**.

### 6.2 Peerness Isolation
Peerness was mathematically derived during Phase 4:
$$\text{Peerness}(u_1, u_2) = \frac{23 - 2 \cdot |\text{Age}_1 - \text{Age}_2|}{30} + 0.23 \cdot \mathbb{I}(\text{School}_1 = \text{School}_2)$$
Because this deterministic demographic score was directly utilized by dataset creators to define ground truth cyberbullying labels, including it as a predictor would artificially inflate accuracy through label leakage.
- In `results/node_sna_features.csv`, peerness is completely absent.
- In `results/dyad_sna_features.csv`, peerness is quarantined under the explicit identifier `exploratory_peerness_score` and marked as prohibited for predictive training.

---

## 7. Verification and Testing Suite

A dedicated unit test module (`tests/test_sna_graph_and_features.py`) was created and integrated into the continuous test runner.

### Test Coverage Summary (8/8 Passed)
1. `test_file_existence`: Verifies all GraphML files, feature CSVs, and predictions exist.
2. `test_node_counts_and_demographics`: Verifies exactly 100 user nodes with complete demographic attributes.
3. `test_edge_counts_and_self_loops`: Confirms 90,356 edges in MultiDiGraph, 9,511 edges in DiGraph, and zero self-loops.
4. `test_prediction_integrity_in_communication`: Asserts no null probabilities and strict $[0.0, 1.0]$ bounds.
5. `test_target_leakage_prevention`: Ensures `CB_Label`, `Aggressive_Count`, and `Intent_to_Harm` are entirely absent from feature sets.
6. `test_node_feature_completeness_and_validity`: Validates completeness, PageRank sum convergence ($= 1.0$), and degree ranges.
7. `test_dyad_feature_completeness_and_validity`: Validates all 21 dyad columns, imbalance ranges $[-1.0, 1.0]$, and message counts $\ge 1$.
8. `test_summary_metrics_consistency`: Validates numerical parity between summary tables and graph structures.

**Full Test Suite Status:**
$$\text{Ran 46 tests across 8 test modules} \implies \mathbf{46\ \text{PASSED},\ 0\ \text{FAILURES},\ 0\ \text{ERRORS}}$$

---

## 8. Summary of Generated Phase 5 Artifacts

| Category | File Path | Records / Size | Description |
| :--- | :--- | :--- | :--- |
| **Pipeline Scripts** | `src/sna/predict_communication.py` | 3.6 KB | Generates calibrated aggression probabilities for communication events |
| | `src/sna/build_graphs.py` | 4.7 KB | Builds and serializes MultiDiGraph and DiGraph in GraphML |
| | `src/sna/feature_engineering.py` | 12.3 KB | Computes node centralities and dyadic interaction features |
| | `src/sna/run_sna_pipeline.py` | 0.7 KB | Orchestrates full Phase 5 execution in sequence |
| **Graph Artifacts** | `results/communication_multidigraph.graphml` | 31.5 MB | 100 nodes, 90,356 directed edges with timestamps & text |
| | `results/communication_digraph.graphml` | 3.0 MB | 100 nodes, 9,511 directed edges with aggregated metrics |
| **Feature Data** | `data/processed/communication_with_predictions.csv` | 18.2 MB | 90,356 rows with calibrated aggression probabilities |
| | `results/node_sna_features.csv` | 15.7 KB | 100 rows $\times$ 22 node-level SNA and demographic features |
| | `results/dyad_sna_features.csv` | 1.45 MB | 9,511 rows $\times$ 21 dyad-level interaction and centrality features |
| | `results/sna_summary.csv` | 1.1 KB | High-level summary metrics of the social communication network |
| **Testing** | `tests/test_sna_graph_and_features.py` | 5.8 KB | 8 automated tests for graph integrity and leakage prevention |
| **Documentation**| `docs/SNA_IMPLEMENTATION_REPORT.md` | This document | Comprehensive technical documentation and verification report |

---

## 9. Conclusion and Readiness for Stage 2

Phase 5 has successfully established a high-integrity, leakage-free graph representation and engineered 43 distinct topological, directional, temporal, and aggression-weighted features.

The project is fully prepared for **Phase 6: Stage 2 Cyberbullying Detection & Evaluation**, wherein these engineered SNA features will be combined with aggression probabilities to model dyadic cyberbullying relationships without target leakage.
