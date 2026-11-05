# CyberShield AI — Machine Learning Final Verification Report
**Component:** Stage 1 — Aggressive vs. Non-Aggressive Language Classification  
**Member:** Member 1 (Artificial Intelligence & Social Network Analysis)  
**Date:** November 5, 2026  
**Status:** Verification Complete — Awaiting Final User Approval to Train  
**Associated Files & Results:**
- [`results/normalized_label_conflicts.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/normalized_label_conflicts.csv) (12 conflicting strings)
- [`data/processed/ml_corpus_conflict_excluded.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_conflict_excluded.csv) (24 excluded rows)
- [`data/processed/ml_corpus_train_split.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_train_split.csv) (89,885 rows, 70.00%)
- [`data/processed/ml_corpus_val_split.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_val_split.csv) (19,258 rows, 15.00%)
- [`data/processed/ml_corpus_test_split.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_test_split.csv) (19,271 rows, 15.01%)
- [`src/ml_data_prep.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml_data_prep.py) (Modular preprocessor & split generator)
- [`tests/test_ml_data_prep.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_ml_data_prep.py) (Automated verification test suite)

---

## 1. Executive Summary

In accordance with user instructions, all four final pre-training checks for the CyberShield AI text classification pipeline have been executed, verified, and tested:

1. **Normalized Label Conflicts:** Inspected all 12 normalized text strings that carry contradictory labels due to capitalization and whitespace variations. Saved details to [`results/normalized_label_conflicts.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/normalized_label_conflicts.csv) and excluded the corresponding 24 rows from the supervised training pool.
2. **Custom URL Normalizer:** Implemented and rigorously tested `cybershield_url_normalizer`. Verified that it converts text to lowercase, maps 6,145 genuine raw and punctuation-stripped URLs to `<url>`, produces zero false positives on ordinary English vocabulary, and preserves critical negation modifiers.
3. **Frequency Reconciliation (The 10,403 Record Difference):** Mathematically resolved and proved why the combined cleaned corpus has 237,585 records while the frequency-weighted clean training total is 227,182. The exact difference of **10,403 records** corresponds to the instances of the 2,756 ambiguous messages isolated during remediation ($7,241 \text{ aggressive} + 3,162 \text{ non-aggressive} = 10,403$).
4. **Reproducible 70/15/15 Data Splitting:** Created a leakage-free split grouped by normalized message (`norm_message`) across 128,414 clean records. Verified **zero normalized message overlap** between Train, Validation, and Test, while preserving the exact 26.95% / 73.05% class balance across all three partitions.
5. **Automated Verification:** All **31 unit tests** across six test suites passed with zero errors.

---

## 2. Check 1 — Normalized Label Conflicts

### 2.1 Problem Analysis
When raw text is normalized (lowercased, punctuation removed, and whitespace collapsed):
$$\text{norm}(t) = \text{strip}(\text{collapse}(\text{remove\_punct}(\text{lower}(t))))$$
Exactly **12 normalized strings** map to conflicting labels (both `label = 1` and `label = 0`).

Because standard TF-IDF vectorizers lowercase text by default, passing both variants into supervised training would map identical feature vectors to opposite binary labels, injecting contradictory gradients into Logistic Regression and Linear SVM loss functions.

### 2.2 Inspection of the 12 Conflicts
All 12 conflicts were isolated and documented in [`results/normalized_label_conflicts.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/normalized_label_conflicts.csv):

| Conflict ID | Normalized Text Sample | Raw Aggressive Variant (`label = 1`) | Raw Non-Aggressive Variant (`label = 0`) | Agg Freq | Non-Agg Freq | Cause of Conflict |
|---|---|---|---|---|---|---|
| **1** | `awesome` | `"Awesome"` | `"awesome"` | 2 | 1 | Capitalization difference |
| **2** | `how old are you` | `"how old are you"` | `"How old are you"` | 2 | 1 | Capitalization difference |
| **3** | `i am not a vandal` | `"I AM NOT A VANDAL"` | `"I am not a vandal"` | 2 | 1 | ALL CAPS vs Title case |
| **4** | `so` | `"So"` | `"so"` | 1 | 1 | Capitalization difference |
| **5** | `let me help you with ur little sock puppet investigation...` | `"let me help you with ur little sock puppet investigation..."` | `"Let me help you with ur little sock puppet investigation..."` | 2 | 1 | Leading sentence capitalization |
| **6** | `glenn first of all the freestylefrappe is not tersely saying...` | Contains `"TERSELY"` | Contains `"TERSELY"` with differing whitespace | 2 | 2 | Internal spacing / line breaks |
| **7** | `i know what your world wide conspiracy you are involved in...` | Paragraph with 2 spaces | Paragraph with 1 space | 6 | 2 | Whitespace discrepancy |
| **8** | `laughing my ass off at attempts to distinguish between...` | `"Laughing my ass off..."` | `"laughing my ass off..."` | 2 | 2 | Capitalization difference |
| **9** | `removing irrelevant nonsense ad hominem attack...` | `"Removing irrelevant..."` | `"removing irrelevant..."` | 2 | 2 | Capitalization difference |
| **10** | `ringerfan i just want you to let you know that you appear...` | `"Ringerfan I just..."` | `"ringerfan I just..."` | 4 | 1 | Capitalization difference |
| **11** | `template talkprussian princes you successfully defended...` | `"Template talkPrussian..."` | `"template talkPrussian..."` | 4 | 2 | Capitalization difference |
| **12** | `your stalker friend is back the hostile user you defended...` | `"Your stalker friend..."` | `"your stalker friend..."` | 4 | 1 | Capitalization difference |

### 2.3 Non-Destructive Remediation
- Neither variant was deleted or arbitrarily relabeled.
- Exactly 24 rows (12 aggressive, 12 non-aggressive, total frequency = 50) were preserved in [`data/processed/ml_corpus_conflict_excluded.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_conflict_excluded.csv).
- They are excluded from the clean training pool ($128,438 - 24 = 128,414$ records).

---

## 3. Check 2 — Custom URL Normalization

### 3.1 Normalizer Implementation
The custom normalizer is implemented in [`src/ml_data_prep.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml_data_prep.py):

```python
URL_REGEX = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b(?:https?|www)[a-z0-9_\-\.\/]{4,}\b"
)

def cybershield_url_normalizer(text, url_token="<url>"):
    text = str(text).lower()
    text = URL_REGEX.sub(url_token, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### 3.2 Verification Results
- **Explicit Lowercasing:** Verified. All output is converted to lowercase.
- **Genuine & Stripped URL Matching:** Matched **6,145 unique URL tokens** across the dataset:
  - Standard URLs: `http://example.com`, `https://en.wikipedia.org` $\to$ `<url>`
  - Punctuation-stripped URLs: `httpstcojXPDgugY`, `httpsenwikipediaorgwikiprimespirals`, `httpmongomutterblogspotcomonedownhtml`, `wwwrleaguecom` $\to$ `<url>`
- **Zero False Positives:** Tested against all tokens under 15 characters matching the regex. No valid English words start with `http` or `www`. Zero ordinary vocabulary words were altered.
- **Negation Preservation:** Explicitly tested and verified that negation words (`not`, `no`, `never`, `don't`, `cannot`, `neither`, `nor`) are 100% retained and untouched.

---

## 4. Check 3 — Frequency Reconciliation (The 10,403 Record Difference)

The user requested an investigation into why the combined cleaned corpus contains **237,585 records**, while the reported frequency-weighted total in `text_corpus_clean_for_training.csv` is **227,182 records**, explaining the difference of **10,403 records**.

### 4.1 Step-by-Step Mathematical Reconciliation

```
+----------------------------------------------------------------------------------------------------+
|                                    FREQUENCY RECONCILIATION AUDIT                                  |
|                                                                                                    |
|  1. Original Raw Datasets:                                                                         |
|     - Aggressive_All.csv:                         118,828 records                                  |
|     - Non_Aggressive_All.csv:                     118,828 records                                  |
|     - Combined Raw Total:                         237,656 records                                  |
|                                                                                                    |
|  2. Blank Text Records Removed:                                                                    |
|     - Aggressive blank records (No. 9152, etc.):       21 records                                  |
|     - Non-Aggressive blank records (No. 1223, etc.):   50 records                                  |
|     - Total Blank Records Removed:                     71 records                                  |
|                                                                                                    |
|  3. Combined Cleaned Corpora:                                                                      |
|     - text_aggressive_cleaned.csv:                118,807 records (118,828 - 21)                   |
|     - text_non_aggressive_cleaned.csv:            118,778 records (118,828 - 50)                   |
|     - Combined Cleaned Corpus Total:              237,585 records (118,807 + 118,778)              |
|                                                                                                    |
|  4. Ambiguous Overlapping Messages Isolated:                                                       |
|     - Unique overlapping text strings:              2,756 messages                                 |
|     - Occurrences in Aggressive Cleaned Corpus:     7,241 records                                  |
|     - Occurrences in Non-Aggressive Cleaned Corpus: 3,162 records                                  |
|     - Total Ambiguous Records Isolated:            10,403 records (7,241 + 3,162)                  |
|                                                                                                    |
|  5. Final Supervised Clean Training Corpus:                                                        |
|     - Total Cleaned Corpus:                       237,585 records                                  |
|     - Minus Isolated Ambiguous Records:          - 10,403 records                                  |
|     - Equals Frequency-Weighted Training Total:   227,182 records                                  |
|                                                                                                    |
|  6. Distinct Deduplication in text_corpus_clean_for_training.csv:                                    |
|     - Sum of 'frequency' column:                  227,182 occurrences                              |
|     - Total unique deduplicated message rows:     128,438 distinct strings                         |
+----------------------------------------------------------------------------------------------------+
```

### 4.2 Exact Conclusion
The difference of **10,403 records** is **100% accounted for**: it is the exact sum of all occurrences of the 2,756 ambiguous overlapping messages across both corpora ($7,241 + 3,162 = 10,403$). When these 10,403 contaminated instances were isolated into [`text_corpus_ambiguous_isolated.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_ambiguous_isolated.csv) to prevent contradictory training gradients, exactly **227,182 unambiguous instances** remained. These 227,182 instances deduplicate cleanly into the **128,438 distinct messages** stored in [`text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv).

---

## 5. Check 4 — Reproducible 70/15/15 Data Splitting

### 5.1 Splitting Methodology
To ensure zero near-duplicate leakage between splits:
1. Started with the 128,414 clean, non-conflicting records.
2. Grouped records by their normalized string (`norm_message`, yielding 128,201 unique groups).
3. Split the groups using stratified partitioning (`random_state=42`) based on the target `label`.
4. Mapped all message variants within a group into the same partition.

### 5.2 Split Proportions & Class Balance

| Split | File Name | Number of Rows | Percentage | Aggressive (`label = 1`) | Non-Aggressive (`label = 0`) | Aggressive Ratio |
|---|---|---|---|---|---|---|
| **Train** | [`ml_corpus_train_split.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_train_split.csv) | 89,885 | 70.00% | 24,218 | 65,667 | **26.94%** |
| **Validation** | [`ml_corpus_val_split.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_val_split.csv) | 19,258 | 15.00% | 5,190 | 14,068 | **26.95%** |
| **Test** | [`ml_corpus_test_split.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/ml_corpus_test_split.csv) | 19,271 | 15.01% | 5,193 | 14,078 | **26.95%** |
| **Total** | — | **128,414** | **100.00%** | **34,601** | **93,813** | **26.95%** |

### 5.3 Overlap Verification
- $\text{Train} \cap \text{Val} = \mathbf{0}$ normalized messages.
- $\text{Train} \cap \text{Test} = \mathbf{0}$ normalized messages.
- $\text{Val} \cap \text{Test} = \mathbf{0}$ normalized messages.
- **Both Logistic Regression and Linear SVM will be evaluated on the exact same split**, enabling rigorous cross-model benchmarking.

---

## 6. Automated Unit Test Results

The comprehensive test suite [`tests/test_ml_data_prep.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_ml_data_prep.py) was executed alongside the existing test suites using `python -m unittest discover tests`:

```text
----------------------------------------------------------------------
Ran 31 tests in 42.063s

OK
```

- **Total Test Suites:** 6
- **Total Tests Run:** 31
- **Passed:** 31
- **Failures:** 0
- **Errors:** 0

All verifications passed: zero missing peerness, 100% chronological ordering, 100% label preservation, 100% referential integrity, complete exclusion of conflicting normalized texts, verified URL normalization without negation loss, exact frequency reconciliation, and zero normalized overlap across ML splits.

---

## 7. Readiness for Model Training

The ML training environment is now verified:
- Clean Train, Validation, and Test sets exist in `data/processed/`.
- Verified preprocessor ready in `src/ml_data_prep.py`.
- No models have been trained yet.

*Awaiting your approval to proceed with training the baseline Logistic Regression and Linear SVM models.*
