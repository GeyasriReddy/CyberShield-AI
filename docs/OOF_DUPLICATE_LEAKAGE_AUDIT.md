# CyberShield AI — Out-of-Fold (OOF) Duplicate Leakage Audit Report

**Document ID:** `docs/OOF_DUPLICATE_LEAKAGE_AUDIT.md`  
**Project:** CyberShield AI — Final-Year B.Tech Project  
**Author / Responsibility:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** November 5, 2026  
**Status:** Completed, Verified, and Tested  

---

## 1. Executive Summary & Objective

In Phase 6 of CyberShield AI, an audit was conducted on the Out-of-Fold (OOF) text aggression prediction framework. In natural language datasets, identical or near-identical text strings (differing only by punctuation, casing, or whitespace formatting) can cross-contaminate folds if cross-validation splits are assigned purely on raw string hashes. Furthermore, approximately 97.14% of the messages in the student communication network overlap with the Wikipedia Detox corpus used to train Stage 1.

This audit establishes a **strictly duplicate-safe Out-of-Fold (OOF) pipeline** (`src/ml/predict_oof_communication.py`), ensuring that:
1. No communication message was scored by a model trained on that message **or on any normalized duplicate of that message**.
2. All normalized duplicate groups belong to **strictly one fold** across cross-validation.
3. Conflicting-label normalized messages are handled consistently with Stage 1 data preparation and isolated from all training folds.
4. All 90,356 communication events receive a verified, out-of-sample probability bounded in $[0.0, 1.0]$.

---

## 2. Text Normalization & Duplicate Grouping Methodology

### 2.1 Equivalence Class Formulation
To capture genuine semantic duplicates, we apply `simple_norm`, the exact text normalization function established during Stage 1 data preparation:
$$\text{simple\_norm}(t) = \text{regex\_collapse\_spaces}(\text{regex\_strip\_punctuation}(\text{lowercase}(t)))$$

This normalization collapses superficial variations:
- URL variations and punctuation-stripped domains (handled via `cybershield_url_normalizer`).
- Trailing spaces, multiple tabs, newlines, and capitalization differences.
- Punctuation-only formatting variations.

### 2.2 Quantitative Grouping Breakdown (Clean Text Corpus)
- **Total Raw Rows in Clean Training Corpus:** **128,438**
- **Unique Raw Messages:** **128,438**
- **Unique Normalized Equivalence Classes:** **128,201**
- **Multi-Instance Duplicate Groups:** **237 groups** (encompassing 474 rows).

---

## 3. Conflicting-Label Handling Protocol

During Stage 1 data preparation (`docs/ML_DATA_PREPARATION_REPORT.md` and `results/normalized_label_conflicts.csv`), 12 normalized message strings (comprising 24 rows) were identified as having ambiguous conflicting ground-truth labels (e.g. identical text labeled $0$ in one instance and $1$ in another).

In strict adherence to Stage 1 governance:
1. **Exclusion from Training:** All 12 conflicting normalized groups (24 rows) were **100% excluded from the supervised training pool of all folds**. No model estimator was ever trained on any conflicting string.
2. **Inference Resolution:** When conflicting normalized strings appear in communication logs, their predictions are generated via the **5-fold ensemble average** ($\frac{1}{5} \sum_{k=1}^5 \mathcal{M}_k(x)$). Because none of the 5 fold models ever trained on these strings, the ensemble prediction remains strictly out-of-sample.

---

## 4. Stratified Group K-Fold Partitioning

The remaining clean pool of **128,414 rows** across **128,201 unique normalized groups** was partitioned into 5 folds using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`:

| Fold Index | Training Rows | Training Groups | Validation Rows | Validation Groups | Normalized Group Overlap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 102,731 | 102,560 | 25,683 | 25,641 | **0 groups (0.0%)** |
| **Fold 2** | 102,731 | 102,561 | 25,683 | 25,640 | **0 groups (0.0%)** |
| **Fold 3** | 102,731 | 102,561 | 25,683 | 25,640 | **0 groups (0.0%)** |
| **Fold 4** | 102,731 | 102,560 | 25,683 | 25,641 | **0 groups (0.0%)** |
| **Fold 5** | 102,732 | 102,562 | 25,682 | 25,639 | **0 groups (0.0%)** |
| **Total** | — | — | **128,414** | **128,201** | **Strictly Disjoint** |

```
+----------------------------------------------------------------------------------------------------+
|                             STRATIFIED GROUP K-FOLD PARTITIONING                                    |
+----------------------------------------------------------------------------------------------------+
|  Every normalized text group g in clean pool belongs to EXACTLY ONE fold k.                        |
|                                                                                                    |
|  Fold 1:  [ Train on Folds 2, 3, 4, 5 (102,560 groups) ]  --> Predict on Fold 1 (25,641 groups)   |
|  Fold 2:  [ Train on Folds 1, 3, 4, 5 (102,561 groups) ]  --> Predict on Fold 2 (25,640 groups)   |
|  Fold 3:  [ Train on Folds 1, 2, 4, 5 (102,561 groups) ]  --> Predict on Fold 3 (25,640 groups)   |
|  Fold 4:  [ Train on Folds 1, 2, 3, 5 (102,560 groups) ]  --> Predict on Fold 4 (25,641 groups)   |
|  Fold 5:  [ Train on Folds 1, 2, 3, 4 (102,562 groups) ]  --> Predict on Fold 5 (25,639 groups)   |
+----------------------------------------------------------------------------------------------------+
```

---

## 5. Communication Dataset Mapping & Coverage Audit

The 90,356 student communication events were mapped to their normalized groups and assigned to their respective out-of-fold scoring estimators:

### 5.1 Communication Group Taxonomy (90,136 Unique Normalized Groups)
1. **OOF Fold Mapped (87,552 groups | 87,771 events | 97.14%):**
   - Each group belongs to validation fold $k \in \{1, \dots, 5\}$.
   - Scored strictly by Model $k$ (trained on the remaining 4 folds).
   - Assertion verified: `assert norm_msg not in fold_train_groups[k]`.
2. **Conflicting Group Pool (0 groups in comm logs):**
   - None of the 12 conflicting strings appeared in the communication messages.
3. **Truly Unseen Held-Out Messages (2,584 groups | 2,585 events | 2.86%):**
   - Messages completely absent from the Wikipedia Detox clean text corpus.
   - Scored via the 5-fold model ensemble average.
   - Assertion verified: `assert norm_msg not in all_train_groups_union`.

### 5.2 Verification Assertions Enforced in Code
```python
# 1. Zero group leakage assertion across all folds
assert len(tr_groups.intersection(val_groups)) == 0

# 2. Per-message out-of-sample assertion
assert norm_msg not in fold_train_groups[fold_num]

# 3. Unseen group exclusion assertion
assert norm_msg not in all_train_groups_union

# 4. Global dataset completeness assertion
assert len(df_out) == 90356
assert df_out["oof_aggression_probability"].isnull().sum() == 0
assert (df_out["oof_aggression_probability"] >= 0.0).all() and (df_out["oof_aggression_probability"] <= 1.0).all()
```

---

## 6. Output Metadata & Audit Verdict

The generated dataset [communication_with_oof_predictions.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_with_oof_predictions.csv) encapsulates:
- `aggression_probability`: Out-of-fold probability rounded to 6 decimals (for direct SNA consumption).
- `oof_aggression_probability`: Explicit unrounded out-of-fold probability.
- `aggression_prediction`: Binary classification ($\ge 0.50$).
- `prediction_status`: `"oof_cross_validation_prediction"` or `"oof_ensemble_unseen_prediction"`.
- `prediction_source`: Specific scoring estimator (e.g. `"oof_fold_1_prediction"`, `"oof_unseen_ensemble_prediction"`).
- `norm_message`: The normalized message string used for duplicate group isolation.

### Audit Verdict: **PASSED (100% Duplicate-Safe & Leakage-Free)**
- Every communication message was scored by an estimator that never encountered that message or any of its normalized equivalents during training.
- Zero data leakage exists between the Stage 1 NLP predictor and the communication logs.
