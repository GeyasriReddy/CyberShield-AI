# CyberShield AI — Stage 1 to Stage 2 Data Overlap & Leakage-Safe OOF Evaluation Report

**Document ID:** `docs/STAGE1_STAGE2_OVERLAP_REPORT.md`  
**Project:** CyberShield AI — Final-Year B.Tech Project  
**Author / Responsibility:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** November 5, 2026  
**Status:** Completed & Empirically Verified  

---

## 1. Executive Summary & Problem Formulation

During the peer audit of Phase 6 (Stage 2 Cyberbullying Detection), a critical research question was identified regarding the boundary between Stage 1 (NLP text aggression classification) and Stage 2 (Dyadic cyberbullying relationship classification):

> **Research Question:** Does the high overlap between the Stage 1 text training corpus and the synthetic communication events used to extract dyadic NLP features result in in-sample feature contamination, and does an out-of-fold (OOF) training strategy preserve Stage 2 predictive validity?

### Key Findings Summary:
1. **Corpus Overlap Quantification:**
   - Total communication events: **90,356** (100% unique textual messages).
   - Cleaned text training corpus: **128,438** unique messages.
   - Text overlap between communication and training corpus: **87,760 messages (97.13%)**.
   - Held-out communication messages not in training corpus: **2,596 messages (2.87%)**.
   - **Seen in Stage 1 Train Split (In-Sample):** **61,546 messages (68.12%)**.
   - **Seen in Stage 1 Val Split (Out-of-Sample):** **13,121 messages (14.52%)**.
   - **Seen in Stage 1 Test Split (Out-of-Sample):** **13,076 messages (14.47%)**.

2. **Root Cause Analysis:**
   The Mendeley cyberbullying dataset authors synthesized the communication logs among the 100 students by drawing messages from the public Wikipedia Detox dataset (`Aggressive_All.csv` and `Non_Aggressive_All.csv`). Because our Stage 1 text classifier was trained on a 70% random stratified split of this same cleaned corpus, 68.12% of the communication messages were technically evaluated "in-sample" when generating baseline dyad features in Phase 5.

3. **Leakage-Safe Resolution via 5-Fold Stratified Out-of-Fold (OOF) Cross-Validation:**
   To guarantee strict out-of-sample validity, we implemented an Out-Of-Fold prediction framework (`src/ml/predict_oof_communication.py`):
   - The 128,438 clean corpus messages were partitioned into 5 stratified folds.
   - Each fold was predicted strictly by an estimator trained on the remaining 4 folds (100% unseen).
   - The 2,596 messages unique to the communication dataset were predicted using an ensemble average across all 5 fold models.
   - The resulting dataset was exported to `data/processed/communication_with_oof_predictions.csv`, leaving the original `data/processed/communication_with_predictions.csv` intact.

4. **Empirical Robustness:**
   - Message-level Pearson correlation between In-Sample and OOF probabilities is **$r = 0.9630$** with a mean absolute difference of only **$0.0904$** ($9.0\%$).
   - Binary decision agreement at threshold $0.50$ is **$92.11\%$**.
   - Dyad-level feature correlations exceed **$0.963$** across all NLP dimensions:
     - `predicted_aggression_volume`: **$r = 0.9680$**
     - `mean_aggression_probability`: **$r = 0.9636$**
     - `max_aggression_probability`: **$r = 0.9670$**
   - Stage 2 cyberbullying classification performance is preserved without artificial inflation.

---

## 2. Quantitative Overlap Breakdown

The table below delineates the distribution of the 90,356 communication events relative to the Stage 1 dataset splits:

| Split / Partition | Split Size (Corpus) | Overlapping Comm Events | % of Total Comm | Stage 1 Status in Phase 5 Baseline | Stage 1 Status in OOF Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 Train Split** | 89,906 (70.0%) | 61,546 | **68.12%** | *In-Sample* (Trained on text) | **Strictly Out-of-Sample** (Fold-held-out) |
| **Stage 1 Val Split** | 19,266 (15.0%) | 13,121 | **14.52%** | Out-of-Sample | **Strictly Out-of-Sample** (Fold-held-out) |
| **Stage 1 Test Split** | 19,266 (15.0%) | 13,076 | **14.47%** | Out-of-Sample | **Strictly Out-of-Sample** (Fold-held-out) |
| **Held-Out (Not in Corpus)**| N/A | 2,596 | **2.87%** | Out-of-Sample | **Strictly Out-of-Sample** (5-Fold Ensemble) |
| **Total Communication** | — | **90,356** | **100.00%** | 68.12% In-Sample / 31.88% OOS | **100.00% Strictly Out-of-Sample** |

```
+---------------------------------------------------------------------------------------+
|                       COMMUNICATION EVENTS (90,356 total events)                      |
+---------------------------------------------------------------------------------------+
|  [================== 68.12% ==================] [== 14.52% ==] [== 14.47% ==] [ 2.87% ] |
|            Seen in Stage 1 Train                  Seen in Val    Seen in Test   Unseen  |
|                                                                                       |
|  Baseline Phase 5:                                                                    |
|  [ In-Sample Prediction                       ] [ Out-of-Sample Prediction            ] |
|                                                                                       |
|  Leakage-Safe 5-Fold OOF:                                                             |
|  [ Fold-Rotated Out-of-Fold Prediction                                                ] |
+---------------------------------------------------------------------------------------+
```

---

## 3. Methodological Comparison: In-Sample vs. Out-of-Fold

### 3.1 Why Linear $L_2$ Regularization Mitigates Severe Overfitting
In deep learning or unregularized non-parametric models, evaluating on training data results in near-zero training error and overconfident probabilities ($p \to 0$ or $p \to 1$). However, our Stage 1 model is:
$$\min_{w} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^N \mathcal{L}(y_i, w^T \phi(x_i))$$
with $C = 1.0$ and 40,000 TF-IDF features across 89,906 training instances. Because $L_2$ regularization shrinks weights towards zero and the sample size is large ($N = 89,906$), the model exhibits minimal variance between training and validation folds.

### 3.2 Mathematical Formulation of 5-Fold OOF Predictions
Let $\mathcal{D}_{\text{corpus}} = \{(x_i, y_i)\}_{i=1}^{128,438}$ be partitioned into 5 disjoint stratified subsets $\mathcal{F}_1, \dots, \mathcal{F}_5$.

For each fold $k \in \{1, \dots, 5\}$:
1. Train pipeline $\mathcal{M}_k = \text{Fit}(\mathcal{D}_{\text{corpus}} \setminus \mathcal{F}_k)$.
2. Predict probabilities on the validation fold:
   $$\hat{p}_i^{\text{OOF}} = \mathcal{M}_k(x_i) \quad \forall x_i \in \mathcal{F}_k$$
   By construction, $(x_i, y_i) \notin \text{Train}(\mathcal{M}_k)$, ensuring $\hat{p}_i^{\text{OOF}}$ is strictly out-of-sample.

For messages $x^* \notin \mathcal{D}_{\text{corpus}}$ (the 2,596 held-out messages):
$$\hat{p}^{\text{OOF}}(x^*) = \frac{1}{5} \sum_{k=1}^5 \mathcal{M}_k(x^*)$$
Because $x^*$ was absent from all training folds, $\hat{p}^{\text{OOF}}(x^*)$ is unconditionally out-of-sample.

---

## 4. Empirical Benchmark & Correlation Analysis

We evaluated the exact quantitative divergence between the Phase 5 baseline predictions and the leakage-safe OOF predictions:

### 4.1 Message-Level Agreement (90,356 Events)
- **Pearson Correlation ($r$):** **$0.96304$**
- **Spearman Rank Correlation ($\rho$):** **$0.95821$**
- **Mean Absolute Error ($\text{MAE}$):** **$0.09039$**
- **Root Mean Squared Difference ($\text{RMSD}$):** **$0.12654$**
- **Binary Decision Agreement ($\tau = 0.50$):** **$92.11\%$** (83,227 / 90,356 events identical)

### 4.2 Dyad-Level Feature Correlations (9,511 Communicating Dyads)
Because dyadic features aggregate message-level probabilities, the central limit effect further dampens individual message variance:

| Feature Name | Feature Dimension | In-Sample Mean | OOF Mean | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: |
| `mean_aggression_probability` | Dyad Aggression Mean | $0.2194$ | $0.2178$ | **$0.96358$** |
| `predicted_aggression_volume` | Dyad Aggression Sum | $2.0864$ | $2.0621$ | **$0.96798$** |
| `max_aggression_probability` | Dyad Aggression Peak | $0.6288$ | $0.6214$ | **$0.96702$** |
| `sender_outgoing_aggression_volume` | Node Aggression Out | $198.44$ | $196.12$ | **$0.99120$** |
| `receiver_incoming_aggression_exposure` | Node Aggression In | $198.44$ | $196.12$ | **$0.98974$** |

---

## 5. Stage 2 Sensitivity Analysis: Impact on Cyberbullying Detection

To test whether Stage 2 performance changes when trained and evaluated on strictly out-of-sample OOF features, we trained identical models on both feature sets using the verified pair-disjoint test set:

| Model Configuration | Architecture | Baseline Test Acc | OOF Test Acc | Baseline Test Prec | OOF Test Prec | Baseline Test Rec | OOF Test Rec | Baseline Test F1 | OOF Test F1 | Baseline Test PR-AUC | OOF Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M2: NLP-Only** | Logistic Regression | $0.8972$ | $0.8923$ | $0.5036$ | $0.4912$ | **$0.9463$** | $0.9329$ | $0.6573$ | $0.6435$ | $0.6157$ | $0.6106$ |
| **M2: NLP-Only** | Random Forest | $0.9035$ | $0.8951$ | $0.5216$ | $0.4981$ | $0.8926$ | $0.8993$ | $0.6584$ | $0.6411$ | $0.6226$ | $0.6336$ |
| **M4: Combined** | Logistic Regression | $0.8986$ | $0.8958$ | $0.5071$ | $0.5000$ | $0.9530$ | **$0.9597$** | $0.6620$ | $0.6575$ | $0.6246$ | $0.6211$ |
| **M4: Combined** | **Random Forest** | **$0.9259$** | **$0.9196$** | **$0.5964$** | **$0.5759$** | $0.8926$ | $0.8658$ | **$0.7151$** | **$0.6917$** | **$0.7033$** | **$0.6917$** |

### Key Observations from Sensitivity Analysis:
1. **Negligible F1 Degradation ($\le 0.023$):**
   When moving from in-sample Stage 1 probabilities to strictly out-of-sample OOF probabilities, Test F1 on the M4 Random Forest drops by only **$-0.0234$** ($0.7151 \to 0.6917$), and PR-AUC drops by only **$-0.0116$** ($0.7033 \to 0.6917$).
2. **Persistent Advantage of SNA Integration:**
   Crucially, in the strictly out-of-sample OOF setting, **M4 Combined Random Forest ($F_1 = 0.6917$, $\text{PR-AUC} = 0.6917$) decisively outperforms M2 NLP-Only ($F_1 = 0.6411$, $\text{PR-AUC} = 0.6336$) by $+5.06\%$ F1 and $+5.81\%$ PR-AUC**.
3. **Linear Model Resilience:**
   For M4 Logistic Regression, the difference between in-sample ($F_1 = 0.6620$) and OOF ($F_1 = 0.6575$) is less than $0.5\%$ ($0.0045$), with recall slightly improving to $95.97\%$.
4. **Conclusion:**
   The performance gain provided by Social Network Analysis is mathematically real and independent of text-level memorization.


---

## 6. Scientific Conclusion & Governance Protocols

1. **Verification of Leakage Safety:**
   The out-of-fold cross-validation framework proves that Stage 2 cyberbullying classification is **not an artifact of in-sample NLP memorization**. The correlation of $r > 0.96$ confirms that regularized TF-IDF models generalize stably to held-out messages.
2. **Artifact Preservation:**
   In compliance with audit rules:
   - Original `communication_with_predictions.csv` remains preserved.
   - Out-of-fold predictions are explicitly saved in `communication_with_oof_predictions.csv`.
   - The Stage 1 model weights (`models/best_model_pipeline.joblib`) remain unaltered.
3. **Reproducibility:**
   The OOF pipeline is reproducible via `src/ml/predict_oof_communication.py`.

---

## 7. Migration Update: Full Out-of-Fold (OOF) Adoption (Option A)

Following formal review of this report, **Option A: Full Out-of-Fold (OOF) Migration** was officially approved and implemented across the entire project pipeline:

1. **Pre-Migration Safety Backup:**
   All 28 baseline artifacts (graphs, features, models, predictions, and test files) were archived to `backups/pre_oof_migration/` with SHA-256 manifest validation.
2. **Duplicate Leakage Fix in OOF Generation:**
   `StratifiedGroupKFold(n_splits=5)` was applied with grouping on `simple_norm` normalized strings, guaranteeing zero cross-fold duplicate text leakage. 12 conflicting strings (24 rows) were isolated from training folds and scored via 5-model ensemble.
3. **Pipeline Migration:**
   - Communication events: `data/processed/communication_with_oof_predictions.csv` (90,356 rows).
   - GraphML graphs: `results/communication_multidigraph.graphml` and `results/communication_digraph.graphml` regenerated with OOF attributes.
   - Features: `data/processed/dyad_sna_features.csv` and `data/processed/node_sna_features.csv` regenerated with OOF probabilities and verified (< $10^{-5}$ error).
   - Models: All 8 Stage 2 models in `models/` retrained strictly on OOF features.
4. **Primary Deployed Pipeline:**
   The OOF pipeline is now the active primary deployed configuration for CyberShield AI, completely free of text memorization leakage. All 62 unit and integration tests pass at 100%.

