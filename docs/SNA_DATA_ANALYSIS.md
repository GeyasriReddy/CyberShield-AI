# CyberShield AI — Phase 4: Social Network Analysis (SNA) Data Inspection Report
**Project:** CyberShield AI (B.Tech Final Year Engineering Project)  
**Member:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** November 5, 2026  
**Status:** Pre-Graph Inspection Complete — Awaiting User Approval to Construct Graph  
**Datasets Inspected:**
- [`data/processed/users_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/users_cleaned.csv) (100 rows, 5 columns)
- [`data/processed/peerness_symmetric.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/peerness_symmetric.csv) (9,900 rows, 3 columns)
- [`data/processed/communication_cleaned_sorted.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_cleaned_sorted.csv) (90,356 rows, 8 columns)
- [`data/processed/cb_labels_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/cb_labels_cleaned.csv) (9,511 rows, 8 columns)

---

## 1. Executive Summary

This report establishes the foundational data inspection, network topology exploration, and mathematical verification required before constructing the Social Network Analysis (SNA) graph in Python using NetworkX. 

### Core Inspection Discoveries:
1. **Network Density & Asymmetry:** The cohort consists of 100 students exchanging 90,356 messages across 9,511 directed sender-receiver dyads. The graph is dense (96.07% density), meaning topological connectivity alone is uninformative; **edge weights, message frequencies, temporal burstiness, and aggression ratios are the critical discriminators**.
2. **Strict Directionality of Cyberbullying:** In 100.00% of cyberbullying dyads (`CB_Label = 1`), the aggression is strictly **one-way (asymmetric)**. In zero instances did both $A \to B$ and $B \to A$ constitute cyberbullying. Consequently, **an undirected graph is fundamentally invalid; the network MUST be modeled as a directed graph (`nx.DiGraph` / `nx.MultiDiGraph`)**.
3. **Exact Mathematical Formula of Peerness:** Peerness is not an abstract concept; we proved with $100\%$ precision that it is deterministically computed from student age difference and school co-enrollment:
   $$\text{Peerness}(u_1, u_2) = \frac{23 - 2 \cdot |\text{Age}_1 - \text{Age}_2|}{30} + 0.23 \cdot \mathbb{I}(\text{School}_1 = \text{School}_2)$$
4. **Target Leakage Alert:** `Aggressive_Count`, `Intent_to_Harm`, and `Total_messages` in `cb_labels_cleaned.csv` are the mathematical building blocks used by the dataset authors to assign `CB_Label`. Using them as input features would constitute severe target leakage. In production, features must be derived dynamically from raw message streams and our Stage 1 ML model predictions.

---

## 2. Dataset Structure & Relational Schema

```
+-----------------------------------------------------------------------------------------------------------------+
|                                          CYBERSHIELD AI RELATIONAL SCHEMA                                       |
|                                                                                                                 |
|  users_cleaned.csv                                                                                              |
|  +----------+-----+--------+-------------+-------+                                                              |
|  | UserID*  | Age | Gender | School Name | Grade |                                                              |
|  +----------+-----+--------+-------------+-------+                                                              |
|       ^                                                                                                         |
|       |                                                                                                         |
|       +------------------------------------+                                                                    |
|       |                                    |                                                                    |
|       | (Sender)                           | (Receiver)                                                         |
|  communication_cleaned_sorted.csv          |                                                                    |
|  +------------+----------+-----------+-----------+---------+-------+                                            |
|  | timestamp  | User1 ID | User2 ID  | Message   | Label   | ...   |                                            |
|  +------------+----------+-----------+-----------+---------+-------+                                            |
|       |                                                                                                         |
|       | Dynamic Edge Aggregation                                                                                |
|       v                                                                                                         |
|  cb_labels_cleaned.csv                     peerness_symmetric.csv                                               |
|  +----------+----------+-------------+     +--------+--------+----------+                                       |
|  | User1 ID | User2 ID | CB_Label**  |     | User 1 | User 2 | Peerness |                                       |
|  +----------+----------+-------------+     +--------+--------+----------+                                       |
|  (* Primary Key; ** Evaluation Target)                                                                          |
+-----------------------------------------------------------------------------------------------------------------+
```

---

## 3. User & Interaction Network Statistics

| Metric | Empirical Value | Contextual / Analytical Meaning |
|---|---|---|
| **Registered User Population** | 100 users | UserIDs contiguous 1 to 100 with zero gaps. |
| **Unique Senders (`User1 ID`)** | 100 users (100%) | Every student has sent at least one message. |
| **Unique Receivers (`User2 ID`)** | 100 users (100%) | Every student has received at least one message. |
| **Isolated Nodes (No Activity)** | **0 users (0.00%)** | Zero inactive nodes in the network. |
| **Total Message Events** | 90,356 events | Interaction timestamps range Jan 2021 – Feb 2022. |
| **Unique Directed Dyads ($u_1 \to u_2$)** | 9,511 dyads | 96.07% of the 9,900 theoretically possible directed pairs. |
| **Unique Undirected Pairs $\{u_1, u_2\}$** | 4,939 pairs | 99.78% of the 4,950 possible undirected pairs. |
| **Reciprocal Communicating Pairs** | 9,144 directed pairs | 4,572 pairs exchange messages in both directions. |
| **One-Way Asymmetric Dyads** | 367 directed pairs | 367 pairs communicate in one direction only. |
| **Self-Interactions ($u_1 == u_2$)** | **0 events (0.00%)** | Zero self-loops present. |
| **Messages per Directed Dyad** | Min: 1, Max: 40, Mean: 9.50, Median: 8.00 | Rich longitudinal interaction history per pair. |
| **Chronological Ordering per Dyad** | **9,511 / 9,511 (100.00%)** | Strictly monotonic timestamps verified in Phase 2. |

### Node Degree & Volume Distributions:
- **Out-Degree (Targets per user):** Mean = 95.1, Min = 86, Max = 99 (Students broadcast widely).
- **In-Degree (Senders per user):** Mean = 95.1, Min = 90, Max = 99 (Students receive widely).
- **Sent Volume (Out-Weight):** Mean = 903.6 messages, Min = 357, Max = 2,015, Std = 424.4 (High variance; hyper-active senders exist).
- **Received Volume (In-Weight):** Mean = 903.6 messages, Min = 731, Max = 1,032, Std = 68.2 (Relatively uniform reception distribution).

---

## 4. Graph Type Recommendation: Directed vs. Undirected

### Recommendation: **Strictly Directed Graph (`nx.DiGraph` & `nx.MultiDiGraph`)**

### Empirical & Sociological Evidence:
1. **100% Cyberbullying Asymmetry:**
   - Out of the 9,511 communicating pairs, **992 dyads are labeled Cyberbullying (`CB_Label = 1`)**.
   - In **zero pairs (0.00%)** was the reverse dyad also labeled cyberbullying:
     $$\text{Mutual Cyberbullying } (A \to B = 1 \land B \to A = 1) = \mathbf{0}$$
   - Cyberbullying is inherently directional: a perpetrator targets a victim. An undirected edge $\{A, B\}$ destroys the perpetrator-victim orientation.
2. **Asymmetric Aggression Ratios:**
   - On reciprocal pairs, user $A$ may send 20 aggressive messages to $B$, while $B$ sends 0 aggressive messages in return. An undirected edge averages these out, completely obscuring the harassment pattern.
3. **MultiDiGraph for Temporal Dynamics:**
   - Because user dyads exchange between 1 and 40 distinct messages across 14 months, a temporal multi-edge graph (`nx.MultiDiGraph`) preserves message timestamps, burstiness, inter-arrival time, and progressive aggression escalation.
   - A collapsed `nx.DiGraph` should be used for computing static node centrality (PageRank, In/Out-degree centrality).

---

## 5. Peerness Data Analysis

### 5.1 Verification of the Mathematical Formulation
We examined [`peerness_symmetric.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/peerness_symmetric.csv) across all 9,900 directed user pairs and correlated it with demographic features from [`users_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/users_cleaned.csv):
- Correlation with Age Difference: $r = -0.9477$
- Correlation with Same School: $r = +0.3268$
- Correlation with Same Gender: $r = 0.0000$ (Gender is not used in peerness).

By testing symbolic regressions, we discovered that **100% of all 9,900 peerness values** match the following exact formula (error $< 10^{-15}$):
$$\text{Peerness}(u_1, u_2) = \frac{23 - 2 \cdot |\text{Age}_1 - \text{Age}_2|}{30} + 0.23 \cdot \mathbb{I}(\text{School}_1 = \text{School}_2)$$

### 5.2 Statistical Properties:
- **Symmetry:** Strictly symmetric ($\text{Peerness}(u_1, u_2) = \text{Peerness}(u_2, u_1)$).
- **Minimum Value:** `0.1000` (Age difference of 10 years, different schools).
- **Maximum Value:** `0.9967` (Same age/grade, same school).
- **Mean:** `0.5359` | **Median:** `0.5667` | **Std:** `0.1854`.
- **Completeness:** 100% of all 9,900 user pairs have a defined peerness score (0 missing values).

### 5.3 Contextual Role of Peerness in Cyberbullying Detection:
In social science and anti-bullying research, peer parity is essential:
- When $\text{Peerness} \approx 1.0$, students are classmates and peer equals.
- When $\text{Peerness} < 0.40$, a major age and institutional disparity exists.
- In this Mendeley dataset, **peer cyberbullying requires $\text{Peerness} > 0.40$**. Dyads with low peerness ($< 0.40$) are classified as non-cyberbullying (`CB_Label = 0`) even if aggressive language is present, because the dataset authors operationalized cyberbullying as **peer-to-peer school harassment**, categorizing cross-school age-disparate arguments as generic external hostility rather than peer cyberbullying.

---

## 6. Cyberbullying Labels & Target Leakage Risks

### 6.1 Distribution of `CB_Label`
- **Class 0 (Non-Cyberbullying Dyads):** 8,519 (89.57%)
- **Class 1 (Cyberbullying Dyads):** 992 (10.43%)
- **Class Imbalance Ratio:** ~8.6 : 1

### 6.2 Target Leakage Identification
We evaluated a shallow Decision Tree on `Total_messages`, `Aggressive_Count`, `Intent_to_Harm`, and `Peerness`. A tree of depth 4 predicts `CB_Label` with **98.03% accuracy**:
- When `Aggressive_Count <= 4`, `CB_Label` is **always 0**.
- When `Peerness <= 0.40`, `CB_Label` is **always 0** (regardless of aggression count).
- When `Aggressive_Count >= 5`, `Peerness > 0.40`, and `Intent_to_Harm > 0.40`, `CB_Label` is **almost always 1**.

### Critical Target Leakage Warning:
- `CB_Label` was synthetically assigned by the dataset creators using a heuristic threshold function of `Total_messages`, `Aggressive_Count`, and `Peerness`!
- **Strict Rule for Member 1:** `Total_messages`, `Aggressive_Count`, and `Intent_to_Harm` from `cb_labels_cleaned.csv` **MUST NEVER be used as input features to a model**. Using them would result in a trivial, circular model that achieves 98%+ accuracy by memorizing the author's heuristic formula rather than learning genuine social network dynamics.
- In production, our model must predict cyberbullying from:
  1. The **predicted aggression probabilities** of our Stage 1 ML model applied to raw text messages.
  2. Structural **NetworkX graph metrics** (in/out-degree centrality, PageRank, reciprocity, burstiness).
  3. Contextual **Peerness** and user demographic attributes.

---

## 7. Compatibility with Completed ML Prediction Pipeline

We inspected [`src/ml/predict.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml/predict.py) and [`models/best_model_pipeline.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/best_model_pipeline.joblib):

1. **Text Preprocessing Alignment:**
   - The ML pipeline uses `cybershield_url_normalizer`, which lowercases text and maps stripped/raw URLs to `<url>`.
   - All 90,356 messages in `communication_cleaned_sorted.csv` are clean, non-null strings compatible with this preprocessor.
2. **Probability Generation:**
   - The deployed model (`CalibratedClassifierCV` over Logistic Regression) provides well-calibrated probabilities via `predict_proba()` (Brier score = `0.0818`).
   - We can annotate each directed communication edge with:
     $$P(\text{aggression} \mid m_t) \in [0.0, 1.0]$$
3. **Ambiguous Messages Handling:**
   - The 2,596 messages flagged with `is_ambiguous_overlap = True` will receive model-predicted aggression probabilities based on their learned vocabulary weights, removing subjective bias.
4. **Data Overlap Awareness:**
   - As documented in Phase 3, 97.13% of communication messages were drawn from the Wikipedia Detox reference pool.
   - When training the Stage 2 graph cyberbullying classifier, we must evaluate on **unseen user nodes / held-out student dyads (GroupKFold by `User1 ID`)** to guarantee that cyberbullying detection generalizability is evaluated without network topology leakage.

---

## 8. Recommended NetworkX Graph Structure

We recommend a **Dual-Layered Graph Architecture**:

```
+----------------------------------------------------------------------------------------------------+
|                                    DUAL-LAYERED NETWORKX ARCHITECTURE                              |
|                                                                                                    |
|  LAYER 1: Temporal Multi-Digraph (nx.MultiDiGraph)                                                 |
|  - Nodes: 100 students (Attributes: Age, Gender, School, Grade).                                   |
|  - Edges: 90,356 individual message events.                                                        |
|  - Edge Attributes: timestamp, message_id, raw_label, predicted_agg_prob, is_ambiguous.            |
|  - Purpose: Temporal burstiness, inter-message response latency, evening vs morning toxicity.      |
|                                                                                                    |
|  LAYER 2: Aggregated Weighted DiGraph (nx.DiGraph)                                                 |
|  - Nodes: 100 students (Enhanced with aggregated SNA centralities).                                |
|  - Edges: 9,511 directed dyads (u1 -> u2).                                                         |
|  - Edge Attributes:                                                                                |
|      * total_messages: count of messages sent from u1 to u2                                        |
|      * total_predicted_aggression: sum of P(aggression) on edge                                    |
|      * mean_aggression_probability: average toxicity score                                         |
|      * peerness_score: symmetric peerness from peerness_symmetric.csv                              |
|      * reciprocity_ratio: ratio of messages u2 sent back to u1                                     |
|      * temporal_span_days: time elapsed between first and last interaction                         |
|  - Purpose: Feature extraction for Stage 2 dyadic Cyberbullying classification (CB_Label).         |
+----------------------------------------------------------------------------------------------------+
```

---

## 9. Suggested Social Network Analysis (SNA) Metrics

For each user node and directed dyad, the following NetworkX metrics should be extracted:

### Node-Level Metrics (Student Behavior Profile):
1. **Weighted Out-Degree ($k_{out}^{w}$):** Total volume of messages transmitted.
2. **Weighted In-Degree ($k_{in}^{w}$):** Total volume of messages received.
3. **Aggressive Out-Degree ($k_{out}^{agg}$):** Total volume of predicted aggressive messages transmitted (Perpetrator propensity).
4. **Aggressive In-Degree ($k_{in}^{agg}$):** Total volume of predicted aggressive messages received (Victim vulnerability).
5. **Directed PageRank:** Identifies structurally prominent individuals in the communication hierarchy.
6. **Hub & Authority Scores (HITS):** Hubs represent users broadcasting heavily across nodes; Authorities represent users heavily cited or targeted.
7. **Betweenness Centrality:** Identifies bridge nodes connecting disparate school cliques.
8. **In/Out Toxicity Ratio:** $\frac{k_{out}^{agg}}{k_{in}^{agg} + 1}$, measuring whether a student is primarily an aggressor or a target.

### Dyad-Level Metrics (Interaction Dynamics):
1. **Message Volume ($N_{u_1 \to u_2}$):** Frequency of contact.
2. **Aggression Density ($\frac{\sum P(agg)}{N_{u_1 \to u_2}}$):** Proportion of toxic interaction.
3. **Reciprocity Asymmetry ($\Delta_{\text{rec}} = \frac{N_{u_1 \to u_2} - N_{u_2 \to u_1}}{N_{u_1 \to u_2} + N_{u_2 \to u_1}}$):** Quantifies one-sided harassment vs mutual conversation.
4. **Peerness Parity ($\mathcal{P}(u_1, u_2)$):** Age and institutional status differential.
5. **Temporal Burstiness ($B$):** Standard deviation of inter-message arrival times (cyberbullying often occurs in intense bursts).

---

## 10. Limitations, Ethical Safeguards & Precautions

1. **Do Not Equate Aggression with Cyberbullying:** High out-degree aggression alone does not prove cyberbullying. In mutual banter among equal peers ($\text{Peerness} \approx 1.0$, balanced reciprocity), aggression is common but non-victimizing. Cyberbullying requires persistent imbalance and unreciprocated hostility.
2. **High Graph Density Limitation:** Because density is 96%, unweighted graph diameter or unweighted degree centrality will be uninformative. All graph algorithms must be **edge-weighted**.
3. **Target Leakage Prohibition:** Never feed `Aggressive_Count` or `Intent_to_Harm` from `cb_labels_cleaned.csv` into feature matrices.
4. **Responsible AI & Privacy:** When processing youth social networks, student IDs must remain pseudonymized, and predictions must provide transparent feature attribution (e.g., distinguishing between banter and systemic harassment).

---

*Report compiled by Member 1 (AI & Social Network Analysis).*  
*SNA Data Inspection Complete. Graph construction will commence upon user review and approval.*
