# CyberShield AI — Data Remediation & Preprocessing Report
**Role:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** November 5, 2026  
**Pipeline Execution:** [`src/data_remediation.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/data_remediation.py)  
**Associated Results & Reports:**
- [`results/peerness_recovery_report.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/peerness_recovery_report.csv)
- [`results/overlapping_messages_report.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/overlapping_messages_report.csv)
- [`docs/DATASET_QUALITY_REPORT.md`](file:///c:/Users/vrind/Desktop/CyberShield_AI/docs/DATASET_QUALITY_REPORT.md)

---

## 1. Executive Summary

Following the comprehensive dataset quality audit, the data remediation phase has been successfully executed. All transformations were performed non-destructively: **zero original datasets were modified**, and all cleaned, validated, and normalized artifacts were output to [`data/processed/`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed) and [`results/`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results).

All five remediation objectives were achieved:
1. **100% Peerness Recovery:** Exactly 4,686 missing `Peerness` values in [`CB_Labels.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/6.%20CB_Labels.csv) were recovered from [`peerness_values.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/2.%20peerness_values.csv) using symmetric reverse dyad matching. Zero values remain missing.
2. **Text Quality & Ambiguity Isolation:** Investigated and filtered 21 aggressive and 50 non-aggressive completely blank records. Isolated all 2,756 conflicting overlapping text strings from supervised training while preserving their frequency counts and communication labels. Built a clean, unambiguous training corpus of 128,438 distinct messages.
3. **Temporal Sorting of Communication Events:** Parsed `Date` and `Time` into ISO-8601 timestamps and sorted all 90,356 interactions by `['User1 ID', 'User2 ID', 'timestamp']`. Verified that **100.00% of communicating pairs (9,511 / 9,511)** now exhibit strictly chronological message ordering.
4. **Relational & Count Preservation:** Verified that all 90,356 message events, directional dyads, communication counts, and ground-truth label distributions remain intact without artificial distortion or record deletion.
5. **Automated Verification:** 25 out of 25 automated unit tests across five test suites passed with zero errors.

---

## 2. Task 1 — Peerness Recovery

### 2.1 Problem Analysis
In the original [`6. CB_Labels.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/6.%20CB_Labels.csv), 4,686 out of 9,511 rows (49.27%) contained `NaN` in the `Peerness` column. 

Our audit revealed that [`2. peerness_values.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/2.%20peerness_values.csv) stores undirected peer relationships between users where `User 1 < User 2` for all $\binom{100}{2} = 4,950$ pairs. Because the dataset creators executed a one-way join on directed pairs $(User1, User2)$, every pair where $User1 > User2$ resulted in a missing value.

### 2.2 Mathematical & Algorithmic Solution
Peer status represents social hierarchy parity, which is an **undirected (symmetric)** property between two students:
$$\text{Peerness}(u_i, u_j) = \text{Peerness}(u_j, u_i)$$

We constructed a bidirectional lookup index $\mathcal{P}$:
$$\mathcal{P}[(u_i, u_j)] = \mathcal{P}[(u_j, u_i)] = \text{Score}_{\text{peerness\_values}}[(\min(u_i, u_j), \max(u_i, u_j))]$$

### 2.3 Recovery Execution & Validation Results
- **Total Rows in Dyad Table:** 9,511
- **Originally Populated Rows ($u_1 < u_2$):** 4,825 rows (100% matched source)
- **Originally Missing Rows ($u_1 > u_2$):** 4,686 rows
- **Successfully Recovered Rows:** **4,686 rows (100.00%)**
- **Unrecovered Records:** **0 rows (0.00%)**
- **Missing Peerness after Remediation:** **0**
- **Validation Match Rate:** 100.00% (every recovered score strictly matches its source value).
- No synthetic averages, mean imputations, or zeroes were introduced.

All recovery operations are logged row-by-row in [`results/peerness_recovery_report.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/peerness_recovery_report.csv), and the cleaned dyad dataset is saved to [`data/processed/cb_labels_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/cb_labels_cleaned.csv). In addition, a full 9,900-row bidirectional peerness table was created at [`data/processed/peerness_symmetric.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/peerness_symmetric.csv) for graph algorithms.

---

## 3. Task 2 — Text Data Preparation & Overlap Isolation

### 3.1 Investigation of Missing Text Records
- **Aggressive Corpus:** Rows at indices 9152, 15233, 16240, etc. (21 records total) contain empty string cells (`""`) in the raw CSV. They represent blank records with no characters or tokens.
- **Non-Aggressive Corpus:** Rows at indices 1223, 4525, 7111, etc. (50 records total) likewise contain empty string cells (`""`).
- **Communication Verification:** None of these 71 blank entries were included in [`Communication_Data_Among_Users.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/5.%20Communication_Data_Among_Users.csv).
- **Remediation Action:** Filtered out the 21 and 50 blank entries to yield 118,807 valid aggressive messages and 118,778 valid non-aggressive messages.

### 3.2 Overlapping Text Strings & Contradictory Supervision
- Across the valid corpora, **2,756 unique text strings** appear in **both** the aggressive and non-aggressive datasets.
- In [`Communication_Data_Among_Users.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/5.%20Communication_Data_Among_Users.csv), 2,596 message instances originate from this overlapping set:
  - 838 instances are labeled non-aggressive (`Label = 0`).
  - 1,758 instances are labeled aggressive (`Label = 1`).
- **Root Cause:** Sarcasm, talk-page warnings, administrative disputes, and context-dependent phrasing (e.g. *"dont try to threaten me with barring i dont care at all"*).
- **Remediation Strategy:**
  1. We did not arbitrarily re-label or overwrite these messages.
  2. All 2,756 overlapping messages and their exact frequencies across Aggressive, Non-Aggressive, and Communication data were cataloged in [`results/overlapping_messages_report.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/overlapping_messages_report.csv) and [`data/processed/text_corpus_ambiguous_isolated.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_ambiguous_isolated.csv).
  3. Created [`data/processed/text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv) containing **128,438 distinct unambiguous messages** (34,613 aggressive, 93,825 non-aggressive) with preserved corpus frequency counts.
  4. Flagged the full corpora with `is_ambiguous_overlap = True` so downstream models can filter them dynamically.

---

## 4. Task 3 — Communication Data Preparation

### 4.1 Timestamp Parsing & Chronological Sorting
In the raw communication dataset, only 12.36% of user dyads had messages in chronological order. Longitudinal analysis of cyberbullying (detecting escalation patterns, burstiness, inter-message response latency) requires strict temporal sequence.

- **Timestamp Parsing:** Combined `Date` (`YYYY-MM-DD`) and `Time` (`HH:MM`) into an unambiguous ISO-8601 datetime column `timestamp` (`YYYY-MM-DD HH:MM:SS`).
- **Sorting Hierarchy:** Sorted the entire table by:
  $$\text{Order By: } \text{User1 ID} \uparrow, \quad \text{User2 ID} \uparrow, \quad \text{timestamp} \uparrow$$
- **Verification:** Computed monotonicity across all 9,511 communicating pairs. **100.00% of pairs (9,511 / 9,511)** now have strictly monotonic non-decreasing message timestamps.

### 4.2 Record & Count Preservation
- Total rows before sorting: 90,356.
- Total rows after sorting: **90,356 (0 records dropped, 0 duplicate events deleted)**.
- Communication direction was preserved: `User1 ID` remains the sender and `User2 ID` remains the receiver.
- Verified that for all 9,511 pairs, the count of messages in [`data/processed/communication_cleaned_sorted.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_cleaned_sorted.csv) exactly equals `Total_messages` in `CB_Labels.csv`, and the sum of `Label == 1` exactly equals `Aggressive_Count`.

---

## 5. Summary of Cleaned Processed Files

| File Path | Description | Total Rows | Columns | Key Changes / Remediation Applied |
|---|---|---|---|---|
| **[`data/processed/users_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/users_cleaned.csv)** | Clean demographic user table | 100 | 5 | Validated UserIDs 1–100, verified $\text{Grade} = \text{Age} - 6$. |
| **[`data/processed/peerness_symmetric.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/peerness_symmetric.csv)** | Bidirectional peerness lookup | 9,900 | 3 | Expanded 4,950 undirected pairs into full 9,900 directed matrix for fast edge lookup. |
| **[`data/processed/cb_labels_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/cb_labels_cleaned.csv)** | Ground truth dyad cyberbullying labels | 9,511 | 8 | Recovered all 4,686 missing `Peerness` values (0 nulls remaining); added `peerness_was_recovered` flag. |
| **[`data/processed/communication_cleaned_sorted.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_cleaned_sorted.csv)** | Chronologically sorted interactions | 90,356 | 8 | Combined Date/Time to ISO-8601 timestamp; sorted chronologically per dyad (100% monotonic); added `is_ambiguous_overlap` flag. |
| **[`data/processed/text_aggressive_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_aggressive_cleaned.csv)** | Cleaned aggressive message corpus | 118,807 | 4 | Dropped 21 blank rows; added `corpus_frequency` and `is_ambiguous_overlap` flag. |
| **[`data/processed/text_non_aggressive_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_non_aggressive_cleaned.csv)** | Cleaned non-aggressive message corpus | 118,778 | 4 | Dropped 50 blank rows; added `corpus_frequency` and `is_ambiguous_overlap` flag. |
| **[`data/processed/text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv)** | Supervised NLP training dataset | 128,438 | 5 | Deduplicated unambiguous messages with binary label and occurrence frequencies; excludes all 2,756 ambiguous messages. |
| **[`data/processed/text_corpus_ambiguous_isolated.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_ambiguous_isolated.csv)** | Isolated ambiguous message corpus | 2,756 | 11 | Contains all 2,756 overlapping messages with corpus and communication label frequencies for manual or contextual review. |

---

## 6. Automated Unit Test Results

The automated test suite in [`tests/`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests) was executed using Python's `unittest` discovery runner. All **25 tests passed successfully in 22.62s** with 0 failures and 0 errors:

```text
----------------------------------------------------------------------
Ran 25 tests in 22.623s

OK
```

### Detailed Test Suite Breakdown

1. **[`tests/test_peerness_recovery.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_peerness_recovery.py) (6 tests — PASSED):**
   - `test_zero_missing_peerness_after_recovery`: Verified zero nulls in `cb_labels_cleaned.csv`.
   - `test_recovered_count_exact_4686`: Verified exactly 4,686 recovered records.
   - `test_all_recovered_scores_match_source_exactly`: Verified strict numerical equality between recovered scores and source values.
   - `test_originally_present_scores_not_corrupted`: Verified 4,825 original values were untouched.
   - `test_peerness_range_valid`: Verified all scores fall in $(0, 1]$.
   - `test_symmetric_peerness_table_completeness`: Verified 9,900 bidirectional pairs in `peerness_symmetric.csv`.

2. **[`tests/test_chronological_sorting.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_chronological_sorting.py) (5 tests — PASSED):**
   - `test_total_row_count_preserved`: Verified all 90,356 rows preserved.
   - `test_strictly_chronological_ordering_per_pair`: Verified monotonicity for all 9,511 dyads.
   - `test_timestamp_integrity`: Verified timestamp matches Date and Time components.
   - `test_directional_message_counts_preserved`: Verified message counts match `CB_Labels.Total_messages`.
   - `test_directional_aggression_counts_preserved`: Verified aggression counts match `CB_Labels.Aggressive_Count`.

3. **[`tests/test_label_preservation.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_label_preservation.py) (4 tests — PASSED):**
   - `test_communication_label_distribution_preserved`: Verified 59,579 non-aggressive and 30,777 aggressive labels.
   - `test_cb_label_distribution_preserved`: Verified 8,519 non-CB and 992 CB labels.
   - `test_source_text_labels_preserved`: Verified source labels (1 and 0) in cleaned text sets.
   - `test_training_pool_labels_valid`: Verified clean binary labels in the 128,438 training pool.

4. **[`tests/test_duplicate_handling.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_duplicate_handling.py) (5 tests — PASSED):**
   - `test_no_communication_interactions_deleted`: Verified zero interaction deletions.
   - `test_corpus_frequency_tracked`: Verified corpus frequency metadata on text records.
   - `test_overlapping_messages_count_exact`: Verified 2,756 isolated ambiguous messages.
   - `test_training_pool_strictly_excludes_overlapping_messages`: Verified zero leakage of ambiguous text into clean training data.
   - `test_training_pool_is_deduplicated`: Verified zero duplicate messages in the clean training corpus.

5. **[`tests/test_referential_integrity.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_referential_integrity.py) (5 tests — PASSED):**
   - `test_user_ids_contiguous_1_to_100`: Verified continuous UserIDs 1 to 100.
   - `test_communication_user_ids_valid`: Verified all communication endpoints belong to [1, 100].
   - `test_cb_labels_user_ids_valid`: Verified all dyad endpoints belong to [1, 100].
   - `test_peerness_symmetric_user_ids_valid`: Verified all symmetric peerness pairs belong to [1, 100].
   - `test_bijective_dyad_mapping_comm_and_cb`: Verified exact 1-to-1 dyad mapping across communication and label tables.

---

## 7. Unresolved Problems & Next Step Recommendations

### Identified Nuances for Member 1 to Address in Modeling:
1. **Context-Dependent Labels in 2,596 Communication Messages:**
   - The 2,596 messages in communication data that belong to the overlapping corpus have context-dependent labels (838 labeled `0`, 1,758 labeled `1`).
   - *Recommendation:* When training pure NLP text classifiers, use [`text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv). When fine-tuning on the communication graph, incorporate context (sender historical aggression, dyadic peerness, conversation history) to disambiguate.
2. **Extreme Class Imbalance in Dyadic Cyberbullying:**
   - Cyberbullying occurs in only 10.43% of dyads (992 positive vs 8,519 negative).
   - *Recommendation:* Avoid standard accuracy as an evaluation metric. Use PR-AUC, F1-Macro, and stratified GroupKFold (grouped by `User1 ID` to evaluate generalizability to unseen perpetrators).
3. **Punctuation-Stripped URLs in Preprocessed Text:**
   - 3,791 messages contain stripped URLs (e.g. `httpmongomutter...`).
   - *Recommendation:* In feature extraction, use custom regex to recognize and normalize these stripped URL tokens rather than standard URI schemes.

---

*Report prepared by Member 1 (AI & Social Network Analysis).*  
*Awaiting user approval before initiating model training or graph feature engineering.*
