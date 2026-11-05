# CYBERSHIELD AI — PHASE 6: FINAL OUT-OF-FOLD (OOF) MIGRATION & COMPLETION REPORT

**Document ID:** `docs/PHASE6_FINAL_OOF_MIGRATION_REPORT.md`  
**Project:** CyberShield AI — B.Tech Final Year Engineering Project  
**Author / Responsibility:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** March 2026  
**Status:** 100% Complete, Formally Audited, and Verified (62/62 Automated Tests Passed)  

---

## 1. Executive Summary & Purpose

This report documents the successful, end-to-end execution of **Option A: Full Out-of-Fold (OOF) Migration** across the entire Phase 6 machine learning and Social Network Analysis (SNA) pipeline of CyberShield AI.

### 1.1 Context & Problem Resolved
In earlier development, approximately 97.13% of the 90,356 synthetic student communication messages overlapped with the Wikipedia Detox text corpus used to train the Stage 1 NLP aggression classifier (with 68.12% located in Stage 1's training split). Furthermore, a text deduplication audit revealed 12 conflicting message strings that occurred in both aggressive and non-aggressive contexts in the raw corpus.

To establish unquestionable scientific rigor and eliminate text-memorization leakage, Option A migrated the entire project to:
1. **Duplicate-Safe 5-Fold Stratified Group Out-of-Fold (OOF) predictions** where normalized duplicate messages are strictly grouped together, preventing cross-fold leakage.
2. **OOF-Derived NetworkX Graphs and SNA Features** where all node and dyad aggression metrics aggregate exclusively out-of-sample predictions.
3. **Retrained Stage 2 Models** where all eight model configurations and the final deployed model (`models/stage2_best_model.joblib`) learn exclusively from leakage-safe OOF features.

All 28 baseline artifacts remain preserved with SHA-256 integrity hashes in `backups/pre_oof_migration/`.

---

## 2. Phase-by-Phase Execution Summary

```
+----------------------------------------------------------------------------------------------------+
|                             FULL OOF MIGRATION EXECUTION LIFECYCLE                                 |
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Backup & SHA-256 Manifest (28 files verified in backups/pre_oof_migration/)               |
|                                       │                                                            |
|  Phase 2: Duplicate Leakage Audit (StratifiedGroupKFold on simple_norm; 12 conflict texts isolated) |
|                                       │                                                            |
|  Phase 3: Generate OOF Communication Predictions (90,356 rows in communication_with_oof_predictions) |
|                                       │                                                            |
|  Phase 4: Migrate SNA Graph & Feature Pipeline (build_graphs.py, feature_engineering.py)           |
|                                       │                                                            |
|  Phase 5: Verify Feature Aggregation (scripts/verify_oof_features.py -> Error < 1e-5; 0 NaNs)       |
|                                       │                                                            |
|  Phase 6: Retrain All 8 Stage 2 Models (M1-M4 x LR, RF; Best model: M4 RF selected by Val F1)       |
|                                       │                                                            |
|  Phase 7: Untouched Test Evaluation (stage2_predictions.csv, confusion matrix, comparison CSV)      |
|                                       │                                                            |
|  Phase 8: Automated Regression Testing (62/62 tests passing in 48.8s)                               |
|                                       │                                                            |
|  Phase 9: Comprehensive Documentation Update (PHASE6_STAGE2_RESULTS.md, OVERLAP_REPORT.md)         |
|                                       │                                                            |
|  Phase 10: Final Handover Package for Member 2 (Dashboard Integration Specifications)               |
+----------------------------------------------------------------------------------------------------+
```

### Phase 1: Pre-Migration Backup & Safety
- **Directory Created:** `backups/pre_oof_migration/`
- **Scope:** 28 total baseline files (4 datasets in `data/processed/`, 9 models in `models/`, 8 result files in `results/`, 2 test files in `tests/`, and 5 documentation files).
- **Verification:** SHA-256 hashes generated for all 28 files and saved in both `manifest.csv` and `manifest.json`. Exact match verified before any migration changes.

### Phase 2: Duplicate Leakage Audit & Cross-Validation Strategy
- **Audit Findings:** Clean text corpus of 128,438 rows groups into 128,201 unique strings under `simple_norm`. 12 normalized strings (24 rows) had conflicting labels (one annotated as aggressive, one as non-aggressive).
- **Resolution:**
  - Applied `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouping by `simple_norm`.
  - Excluded the 12 conflict strings from all 5 training folds (matching Stage 1 cleaning protocols).
  - Out-of-fold predictions generated for all 87,751 mapped communication events.
  - The 20 conflict events and 2,585 unseen communication events were scored using an unbiased 5-model ensemble average.
  - Zero cross-fold group overlap verified across all 5 folds.
- **Audit Documentation:** `docs/OOF_DUPLICATE_LEAKAGE_AUDIT.md`.

### Phase 3: Out-of-Fold Communication Predictions
- **Script Executed:** `src/ml/predict_oof_communication.py`
- **Output Artifact:** `data/processed/communication_with_oof_predictions.csv`
- **Integrity Validation:** Exactly 90,356 records, 0 nulls/NaNs, probabilities bounded strictly in $[0.000065, 1.000000]$ (mean: $0.401169$).
- **Aggressive Message Count ($\tau \ge 0.50$):** 32,504 events.

### Phase 4: SNA Pipeline Migration & GraphML Construction
- **Scripts Updated:**
  - `src/sna/build_graphs.py`: Defaults to `communication_with_oof_predictions.csv`, adds OOF edge attributes.
  - `src/sna/feature_engineering.py`: Synchronizes exports to both `results/` and `data/processed/`.
  - `src/sna/run_sna_pipeline.py`: Executes end-to-end graph construction and feature engineering.
- **Graph Artifacts:**
  - `results/communication_multidigraph.graphml` (Temporal MultiDiGraph: 100 nodes, 90,356 edges, 0 self-loops).
  - `results/communication_digraph.graphml` (Aggregated DiGraph: 100 nodes, 9,511 directed dyads, density $0.9607$, 0 self-loops).

### Phase 5: Independent Feature Aggregation Verification
- **Script Executed:** `scripts/verify_oof_features.py`
- **Verification Results:**
  - `data/processed/` vs `results/` parity: **100% Identical** (`assert_frame_equal` passed).
  - Dyad-level aggregation error against direct recalculation: $< 10^{-5}$ across `mean_aggression_probability`, `predicted_aggression_volume`, and `max_aggression_probability`.
  - Node-level aggregation error against direct recalculation: $< 10^{-5}$ for incoming and outgoing aggression volumes.
  - Data Quality: Zero NaNs, zero infinite values across all 20 predictive dyad features.
- **Verification Report:** `docs/OOF_SNA_FEATURE_VERIFICATION.md`.

### Phase 6: Retraining All Eight Stage 2 Models
- **Script Executed:** `src/ml/train_stage2.py`
- **Splits:** 70% Train (6,662 dyads), 15% Val (1,419 dyads), 15% Test (1,430 dyads) grouped by unordered student pair (`pair_id = min(u,v)_max(u,v)`), guaranteeing zero reverse-dyad leakage.
- **Models Fitted & Exported:**
  1. `models/stage2_m1_baseline_logistic_regression.joblib` (6 features)
  2. `models/stage2_m1_baseline_random_forest.joblib` (6 features)
  3. `models/stage2_m2_nlp_only_logistic_regression.joblib` (3 features)
  4. `models/stage2_m2_nlp_only_random_forest.joblib` (3 features)
  5. `models/stage2_m3_sna_only_logistic_regression.joblib` (13 features)
  6. `models/stage2_m3_sna_only_random_forest.joblib` (13 features)
  7. `models/stage2_m4_combined_logistic_regression.joblib` (18 features)
  8. `models/stage2_m4_combined_random_forest.joblib` (18 features)
  9. `models/stage2_best_model.joblib`: Selected strictly by Validation F1 (**M4 Combined Random Forest**, Val F1: $0.7228$).
- **Explainability:** Feature importance extracted dynamically per architecture (Gini impurity for Random Forest, absolute standardized coefficients for Logistic Regression) and saved to `results/stage2_feature_importance.csv`.

### Phase 7: Untouched Test Set Evaluation
- **Script Executed:** `src/ml/evaluate_stage2.py`
- **Artifacts Updated:** `results/stage2_predictions.csv`, `results/stage2_confusion_matrix.png`, `results/stage2_model_comparison.csv`.
- **Primary Deployed Model (M4 Random Forest on Untouched Test Set):**
  - **Accuracy:** $91.33\%$ (1,306 / 1,430 correct)
  - **Precision:** $55.32\%$ (130 TP / 235 predicted positive)
  - **Recall:** $87.25\%$ (130 TP / 149 ground-truth positive)
  - **F1-Score:** $0.6771$
  - **Macro F1:** $0.8135$
  - **PR-AUC:** $0.7012$
  - **ROC-AUC:** $0.9635$
  - **Confusion Matrix:** TN: 1,176 | FP: 105 | FN: 19 | TP: 130

### Phase 8: Automated Regression Testing
- **Command:** `python -m unittest discover -s tests -v`
- **Result:** **62 tests passed, 0 failures, 0 errors** (executed in 48.811s).
- **Test Modules Passing:**
  - `tests/test_clean_data.py`: 12/12 passed
  - `tests/test_referential_integrity.py`: 5/5 passed
  - `tests/test_peerness_recovery.py`: 6/6 passed
  - `tests/test_ml_training_and_inference.py`: 10/10 passed
  - `tests/test_sna_graph_and_features.py`: 8/8 passed (including OOF prediction sources & aggressive count checks)
  - `tests/test_stage2_training_and_evaluation.py`: 15/15 passed (including OOF feature distribution, target leakage prevention, pair isolation, and artifact loading checks)
  - Other unit tests: 6/6 passed

### Phase 9: Documentation Updates
- Updated `docs/PHASE6_STAGE2_RESULTS.md` with complete OOF benchmarks, updated tables, and 62/62 test status.
- Updated `docs/STAGE1_STAGE2_OVERLAP_REPORT.md` with Section 7 documenting the successful execution of Option A.
- Updated `docs/PHASE6_CODE_REVIEW.md` updating scorecard to 22/22 items passing and 62 tests.

---

## 3. Comprehensive Performance Comparison: Baseline vs. OOF

The following table presents the complete comparison between the baseline in-sample pipeline and the newly deployed out-of-fold pipeline across all 8 configurations on the untouched test split (1,430 dyads):

| Configuration | Architecture | Baseline Test F1 | OOF Deployed Test F1 | Baseline Test PR-AUC | OOF Deployed Test PR-AUC | Baseline Test Recall | OOF Deployed Test Recall | Baseline Test Acc | OOF Deployed Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Interaction Baseline** | Logistic Regression | $0.6165$ | **$0.6165$** | $0.5728$ | **$0.5728$** | $0.8792$ | **$0.8792$** | $0.8860$ | **$0.8860$** |
| **M1: Interaction Baseline** | Random Forest | $0.6443$ | **$0.6443$** | $0.6301$ | **$0.6301$** | $0.8389$ | **$0.8389$** | $0.9035$ | **$0.9035$** |
| **M2: NLP-Only** | Logistic Regression | $0.6573$ | **$0.6465$** | $0.6157$ | **$0.6101$** | $0.9463$ | **$0.9329$** | $0.8972$ | **$0.8937$** |
| **M2: NLP-Only** | Random Forest | $0.6584$ | **$0.6415$** | $0.6226$ | **$0.6169$** | $0.8926$ | **$0.9128$** | $0.9035$ | **$0.8937$** |
| **M3: SNA-Only** | Logistic Regression | $0.5991$ | **$0.5991$** | $0.5712$ | **$0.5712$** | $0.8926$ | **$0.8926$** | $0.8755$ | **$0.8755$** |
| **M3: SNA-Only** | Random Forest | $0.6560$ | **$0.6560$** | $0.6485$ | **$0.6485$** | $0.8255$ | **$0.8255$** | $0.9098$ | **$0.9098$** |
| **M4: Combined (NLP + SNA)** | Logistic Regression | $0.6620$ | **$0.6559$** | $0.6246$ | **$0.6207$** | $0.9530$ | **$0.9530$** | $0.8986$ | **$0.8958$** |
| **M4: Combined (NLP + SNA)** | **Random Forest (Best)** | **$0.7151$** | **$0.6771$** | **$0.7033$** | **$0.7012$** | **$0.8926$** | **$0.8725$** | **$0.9259$** | **$0.9133$** |

### Key Scientific Takeaways:
1. **Persistent SNA Advantage Under Strict Zero-Leakage:**
   Even when all NLP features are computed out-of-fold, **M4 Combined Random Forest ($F_1 = 0.6771$, $\text{PR-AUC} = 0.7012$) decisively outperforms M2 NLP-Alone ($F_1 = 0.6415$, $\text{PR-AUC} = 0.6169$) by $+3.56\%$ F1 and $+8.43\%$ PR-AUC**.
2. **Precision Enhancement:**
   Social Network Analysis directional features (`directional_communication_imbalance`, `reciprocity_ratio`) and structural centrality (`sender_hub_score`, `sender_out_strength`) reduce false alarms by filtering out heated but symmetric peer squabbles, boosting precision from $49.45\%$ (NLP-alone) to $55.32\%$ (Combined).
3. **PR-AUC Resilience:**
   While F1 shows minor natural shrinkage due to out-of-sample confidence smoothing, Precision-Recall AUC remains virtually unchanged ($0.7033 \to 0.7012$), proving that ranking quality and discrimination are fully preserved.

---

## 4. Top Feature Rankings (M4 Random Forest)

From `results/stage2_feature_importance.csv`:

| Rank | Feature Name | Gini Importance | Behavioral Role |
| :---: | :--- | :---: | :--- |
| **1** | `predicted_aggression_volume` | **0.267841** | Cumulative toxic probability mass transmitted $u_1 \to u_2$. |
| **2** | `total_messages` | **0.160288** | Overall communication frequency (repetition requirement). |
| **3** | `sender_outgoing_aggression_volume` | **0.118141** | Habitual aggression across all peers in the cohort. |
| **4** | `sender_hub_score` | **0.081275** | Prominence as an outgoing communication broadcaster. |
| **5** | `sender_out_strength` | **0.065820** | Outgoing interaction strength / conversational dominance. |
| **6** | `max_aggression_probability` | **0.064501** | Peak single-event threat severity. |
| **7** | `inter_message_time_mean_hours` | **0.056690** | Temporal spacing (rapid messaging bursts indicate harassment). |
| **8** | `mean_aggression_probability` | **0.049001** | Baseline toxicity average. |
| **9** | `interaction_span_days` | **0.030061** | Duration of contact over time. |
| **10**| `directional_communication_imbalance` | **0.022031** | One-way communication asymmetry. |
| **11**| `reciprocity_ratio` | **0.021430** | Ratio of replies received from victim. |
| **12**| `inter_message_time_std_hours` | **0.016320** | Temporal irregularity/jitter. |
| **13-18**| Reciprocal Msgs & Receiver Metrics | **$< 0.010000$** | Receiver centralities have minor relative impact. |

---

## 5. Member 2 Handover Package (Dashboard & Backend Integration)

This section provides complete specifications for **Member 2** (Web Application & Dashboard Integration) to load, query, and visualize Stage 2 cyberbullying predictions.

### 5.1 Primary Model Artifact Details
- **File Location:** `models/stage2_best_model.joblib`
- **File Size:** ~6.1 MB
- **Model Loader:** Python `joblib.load("models/stage2_best_model.joblib")`
- **Bundle Keys:**
  ```python
  bundle = joblib.load("models/stage2_best_model.joblib")
  pipeline = bundle["pipeline"]               # scikit-learn Pipeline (StandardScaler -> RandomForestClassifier)
  feature_names = bundle["feature_names"]     # List of exactly 18 feature names in required order
  model_name = bundle["model_name"]           # "M4_Combined (Random Forest)"
  architecture = bundle["architecture"]       # "Random Forest"
  features_config = bundle["features_config"] # "M4_Combined"
  num_features = bundle["num_features"]       # 18
  probability_type = bundle["probability_type"] # "predicted_probability_uncalibrated"
  val_metrics = bundle["val_metrics"]         # Dict of validation metrics (val_f1: 0.7228)
  ```

### 5.2 Required Input Feature Schema (18 Features in Exact Order)
When querying `pipeline.predict_proba(X)` or `pipeline.predict(X)`, pass a DataFrame containing these 18 columns:

```python
REQUIRED_FEATURES = [
    "total_messages",                        # int: Total messages sent from u1 to u2
    "reciprocal_messages",                  # int: Messages sent in reverse from u2 to u1
    "reciprocity_ratio",                    # float [0.0, 1.0]: n_reverse / total_messages_both_ways
    "directional_communication_imbalance",  # float [-1.0, 1.0]: (forward - reverse) / (total + 1e-6)
    "interaction_span_days",                # float: Duration in days between first and last message
    "inter_message_time_mean_hours",        # float: Average hours between consecutive forward messages
    "inter_message_time_std_hours",         # float: Std dev of hours between consecutive forward messages
    "mean_aggression_probability",          # float [0.0, 1.0]: Average OOF aggression probability
    "predicted_aggression_volume",          # float: Sum of OOF aggression probabilities
    "max_aggression_probability",           # float [0.0, 1.0]: Maximum OOF aggression probability
    "sender_outgoing_aggression_volume",    # float: Sender total aggression sent across all cohort peers
    "receiver_incoming_aggression_exposure",# float: Receiver total aggression received across all peers
    "sender_out_strength",                  # int: Total outgoing messages sent by u1 in network
    "receiver_in_strength",                 # int: Total incoming messages received by u2 in network
    "sender_pagerank",                      # float: PageRank centrality of sender
    "receiver_pagerank",                    # float: PageRank centrality of receiver
    "sender_hub_score",                     # float: HITS hub score of sender
    "receiver_authority_score"              # float: HITS authority score of receiver
]
```

> [!IMPORTANT]
> **Column Re-Ordering Safety:** Always index the incoming DataFrame with `df[bundle["feature_names"]]` before calling `pipeline.predict_proba()`. The pipeline contains a built-in `StandardScaler` fitted on this exact column order.

### 5.3 Output Schema & Probability Interpretation
- **`cyberbullying_probability`:** Raw predicted decision probability $P \in [0.0, 1.0]$ representing tree voting proportion.
- **Decision Threshold:** $\tau = 0.50$ (Standard threshold for binary classification).
- **Recommended Risk Tiers for Dashboard UI:**
  - 🔴 **High Cyberbullying Risk (Alert / Case File):** $P \ge 0.75$ (Immediate counselor notification)
  - 🟡 **Moderate / Monitored Watchlist:** $0.50 \le P < 0.75$ (Flagged for contextual review)
  - 🟢 **Low / Non-Bullying Interaction:** $P < 0.50$ (Normal communication)

### 5.4 Graph Artifacts for Dashboard Network Visualizations
For rendering interactive network graphs on the dashboard (e.g. via D3.js, Cytoscape, or PyVis):
- **Full Communication MultiDiGraph:** `results/communication_multidigraph.graphml` (Contains all 90,356 individual message edges with timestamps and OOF probabilities).
- **Aggregated Dyadic DiGraph:** `results/communication_digraph.graphml` (Contains 9,511 directed dyad edges with aggregated volumes, span, and aggression weights).
- **Node Demographics & Centralities:** `data/processed/node_sna_features.csv` (100 students: UserID, Age, Gender, School, Grade, PageRank, Hub/Authority, Aggression volumes).
- **Dyad Features & Predictions:** `data/processed/dyad_sna_features.csv` and `results/stage2_predictions.csv`.

---

## 6. Final Phase 6 Completion Verdict

- **Option A Full OOF Migration:** **100% COMPLETE & OPERATIONAL**
- **Referential & Feature Integrity:** **100% VERIFIED (< 1e-5 error)**
- **Target Leakage:** **STRICTLY ZERO (CB_Label, Aggressive_Count, Intent_to_Harm, Peerness excluded)**
- **Pair Isolation:** **STRICTLY ENFORCED (Zero reverse-dyad leakage across train/val/test)**
- **Automated Tests:** **62 / 62 PASSED (100% SUCCESS RATE)**
- **Phase 7 (Dashboard Integration) Readiness:** **OFFICIALLY APPROVED FOR HANDOVER**
