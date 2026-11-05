# CyberShield AI — Machine Learning Text Classification Preparation Report
**Component:** Stage 1 — Aggressive vs. Non-Aggressive Language Classification  
**Member:** Member 1 (Artificial Intelligence & Social Network Analysis)  
**Date:** November 5, 2026  
**Status:** Pre-Modeling Inspection Complete — Awaiting User Approval to Train  
**Source Files Inspected:**
- [`data/processed/text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv) (128,438 rows)
- [`data/processed/text_corpus_ambiguous_isolated.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_ambiguous_isolated.csv) (2,756 rows)
- [`data/processed/cb_labels_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/cb_labels_cleaned.csv) (9,511 rows)
- [`data/processed/communication_cleaned_sorted.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_cleaned_sorted.csv) (90,356 rows)

---

## 1. Executive Summary

This report establishes the data preparation, feature engineering strategy, and data leakage safeguards for our **first machine learning model**: an NLP text classifier that distinguishes between **aggressive language (`1`)** and **non-aggressive language (`0`)**.

### Critical Distinctions Clarified:
- **Aggression (Message-Level):** A text classification task detecting toxic, offensive, abusive, or hostile language within an individual message.
- **Cyberbullying (Dyad-Level):** A longitudinal sociotechnical phenomenon requiring repetition, power disparity (peerness), and intent to harm between a specific pair of users over time.
- **Model Role:** Our Stage 1 model predicts **aggression**, not cyberbullying. Its predictions and aggression probabilities will subsequently feed into Stage 2 (Social Network Analysis & Cyberbullying Detection).

---

## 2. Dataset Structure & Column Identification

### 2.1 File-by-File Column Inspection

| Dataset | Text Column | Label Column | Target Meaning | Role in Pipeline |
|---|---|---|---|---|
| **[`text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv)** | `message` (`object`) | `label` (`int64`: 0 or 1) | Aggressive (`1`) vs Non-Aggressive (`0`) | **Primary NLP Training Dataset** |
| **[`text_corpus_ambiguous_isolated.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_ambiguous_isolated.csv)** | `message` (`object`) | `ambiguity_status` (`object`) | Conflicting source labels across corpora | **Isolated from Supervised Training** |
| **[`cb_labels_cleaned.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/cb_labels_cleaned.csv)** | *None* (Dyad summary) | `CB_Label` (`int64`: 0 or 1) | Cyberbullying Dyad (`1`) vs Non-CB (`0`) | **Stage 2 Target Only (Do Not Use for Text Model)** |
| **[`communication_cleaned_sorted.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_cleaned_sorted.csv)** | `Message` (`object`) | `Label` (`int64`: 0 or 1) | Ground-truth message aggression in graph | **Social Network Communication Graph** |

### 2.2 Schema of the Clean Training Dataset
- `message`: The raw text of the comment/message.
- `label`: Binary classification target (`1` = Aggressive, `0` = Non-Aggressive).
- `frequency`: Integer count indicating how many times this exact text appeared in the original Wikipedia Detox corpus.
- `char_length`: Number of characters in the message.
- `word_count`: Number of whitespace-separated words in the message.

---

## 3. Class Distribution & Frequency Dynamics

### 3.1 Distinct Message Distribution (Vocabulary Diversity)
When deduplicated to unique text strings, the dataset exhibits a moderate class imbalance:
- **Class 0 (Non-Aggressive):** 93,825 unique messages (**73.05%**)
- **Class 1 (Aggressive):** 34,613 unique messages (**26.95%**)
- **Total Unique Messages:** 128,438
- **Imbalance Ratio:** ~2.71 : 1

### 3.2 Frequency-Weighted Distribution (Empirical Prevalence)
When accounting for the occurrence frequency of each message (`frequency` column representing all 227,182 original instances):
- **Class 0 (Non-Aggressive Total Instances):** 115,616 (**50.89%**)
- **Class 1 (Aggressive Total Instances):** 111,566 (**49.11%**)
- **Imbalance Ratio:** ~1.04 : 1 (Nearly perfectly balanced!)

### 3.3 Sociolinguistic Explanation
Why is the distinct count imbalanced (73% non-aggressive) while the instance count is balanced (51% vs 49%)?
- **Aggressive messages repeat frequently:** Hostile users repeatedly employ identical insults, profanity, and harassment boilerplate (68.55% repetition rate in the raw aggressive corpus).
- **Non-aggressive messages are linguistically diverse:** Polite conversation, encyclopedic debates, and greetings utilize a much wider vocabulary with lower repetition (18.72% repetition rate).

---

## 4. Verification of Ambiguity Isolation

- **Verification Result:** The intersection between [`text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv) and [`text_corpus_ambiguous_isolated.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_ambiguous_isolated.csv) is strictly **0 messages (0.00%)**.
- All **2,756 conflicting messages** identified in our audit remain safely isolated. No text string that was labeled both aggressive and non-aggressive in the source corpora is permitted to introduce contradictory gradients into supervised training.

---

## 5. Normalized Duplicates & Case/Punctuation Inconsistencies

While raw string deduplication isolated all exact overlaps, standard NLP tokenizers normalize text by lowercasing and stripping punctuation. We performed a deep normalization check:

$$\text{Normalize}(t) = \text{strip}(\text{collapse\_spaces}(\text{remove\_punct}(\text{lower}(t))))$$

### 5.1 Findings
- **Normalized Duplicates:** 225 messages (0.18%) become identical after lowercasing and punctuation stripping.
- **Conflicting Normalized Texts:** Exactly **12 normalized text strings** map to conflicting labels (both 0 and 1).
- **Representative Examples:**
  - `'awesome'`: raw `'Awesome'` was in aggressive (`label = 1`, freq 2); raw `'awesome'` was in non-aggressive (`label = 0`, freq 1).
  - `'how old are you'`: raw `'how old are you'` (`label = 1`) vs `'How old are you'` (`label = 0`).
  - `'i am not a vandal'`: raw `'I AM NOT A VANDAL'` (`label = 1`) vs `'I am not a vandal'` (`label = 0`).
  - Several long talk-page paragraphs differing only by extra internal spacing or capitalized words.

### 5.2 Recommendation
Before fitting vectorizers, group or filter these 12 conflicting normalized pairs to ensure that no lowercased text string exists in both classes.

---

## 6. URL Formatting & Artifact Inspection

### 6.1 Punctuation-Stripped URLs
During earlier benchmark preparation, URLs underwent aggressive punctuation stripping:
- Standard URL patterns (`https?://` or `www\.`) match **0 messages**.
- Custom regex (`\b(?:https?|www)[a-zA-Z0-9_\-\.\/]+`) reveals **5,634 messages (4.39%)** containing stripped URLs (e.g., `httpstcojXPDgugY`, `httpsenwikipediaorgwikiprimespirals`, `httpenwikipediaorgwikiUserRaul`).

### 6.2 Anonymization & Formatting Tokens
- **`NAME` mask token:** Appears in 75 messages (e.g., `"Hello NAME, stop reverting"`).
- **`IP` mask token:** Appears in 1,624 messages (e.g., `"That IP is blocked"`).
- **ALL CAPS (Yelling):** 1,805 messages in the clean training set are typed entirely in capital letters.

### 6.3 Tokenization Strategy
If left untreated, standard tokenizers will split `httpsenwikipediaorgwikiprimespirals` into random character fragments or out-of-vocabulary junk.
- **Preprocessor Rule:** Replace any token matching `\b(?:https?|www)[a-zA-Z0-9_\-\.\/]+` with a unified special token: `<URL>`.
- **Mask Tokens:** Retain `<NAME>` and `<IP>` as explicit vocabulary tokens, as their presence often correlates with Wikipedia administrative interactions.

---

## 7. Data Leakage Analysis & Guardrails

We examined two potential vectors of data leakage:

### 7.1 Cross-Dataset Overlap with Communication Data
- **Finding:** **87,760 out of 90,356 messages (97.13%)** in [`communication_cleaned_sorted.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/communication_cleaned_sorted.csv) are identical to messages in [`text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv).
- **Label Consistency:** 100% agreement (all 87,760 overlapping messages have identical binary labels in both datasets; 0 mismatches).
- **Implication for Member 1:**
  - For evaluating the Stage 1 text classifier, internal Train/Val/Test splits inside `text_corpus` provide an unbiased assessment of aggressive language classification.
  - However, when evaluating the **Stage 2 Cyberbullying pipeline** on test user dyads, the model must not be evaluated as if the communication messages were unseen if the text classifier has already trained on them.
  - *Guardrail:* The Stage 1 text model should be treated as a **pre-trained domain feature extractor** that produces message-level aggression scores $P(\text{aggression} \mid m_t)$. When evaluating Stage 2, user dyads should be partitioned using GroupKFold so that generalizability is assessed on unseen network topologies.

### 7.2 Train/Validation/Test Split Leakage
- To prevent near-duplicate leakage across splits, splitting must be performed **after grouping by normalized text** (`norm_message`). This ensures that case-variations of the same comment never appear in both Train and Test.

---

## 8. Recommended Training, Validation & Testing Splits

We recommend a **Stratified 70% / 15% / 15% Split**:

| Split | Percentage | Number of Messages | Aggressive (Class 1) | Non-Aggressive (Class 0) | Purpose |
|---|---|---|---|---|---|
| **Training** | 70% | ~89,906 | ~24,229 (26.95%) | ~65,677 (73.05%) | Model parameter optimization |
| **Validation** | 15% | ~19,266 | ~5,192 (26.95%) | ~14,074 (73.05%) | Hyperparameter tuning & threshold calibration |
| **Testing** | 15% | ~19,266 | ~5,192 (26.95%) | ~14,074 (73.05%) | Final unbiased performance benchmark |
| **Total** | 100% | 128,438 | 34,613 (26.95%) | 93,825 (73.05%) | Full clean dataset |

### Splitting Protocol:
1. Set a fixed random seed: `random_state = 42`.
2. Stratify on the target `label` column to preserve exact class ratios.
3. Group by `norm_message` to enforce zero normalized duplicate leakage.

---

## 9. Handling Repeated Messages & Frequencies

In [`text_corpus_clean_for_training.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/data/processed/text_corpus_clean_for_training.csv), each row is a distinct unique string with an associated `frequency` value. We evaluated three training approaches:

```
+-----------------------------------------------------------------------------------------+
|                                    TRAINING APPROACHES                                  |
|                                                                                         |
|  1. Unweighted Unique Training (Recommended for Baseline)                               |
|     - Train on 128,438 unique rows with class_weight='balanced'.                         |
|     - Prevents model from overfitting to repetitive insults; maximizes lexical diversity.|
|                                                                                         |
|  2. Log-Dampened Sample Weighting (Recommended for Comparison)                          |
|     - Sample weight: w_i = 1.0 + log(frequency_i).                                      |
|     - Rewards frequent conversational patterns without letting single spam phrases      |
|       dominate loss.                                                                    |
|                                                                                         |
|  3. Full Row Expansion (Not Recommended)                                                |
|     - Expand to 227,182 rows. Wasteful computationally; high duplicate overfitting risk. |
+-----------------------------------------------------------------------------------------+
```

### Recommendation:
Train baseline classifiers (Logistic Regression, LinearSVC, MultinomialNB) on the **distinct unique dataset** with `class_weight='balanced'`. Evaluate whether log-frequency sample weighting improves validation PR-AUC.

---

## 10. Recommended TF-IDF Feature Extraction Settings

Based on empirical testing of vocabulary density and n-gram coverage, the following `TfidfVectorizer` configuration is recommended:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def cybershield_text_preprocessor(text):
    # 1. Normalize punctuation-stripped and raw URLs to unified token
    text = re.sub(r"\b(?:https?|www)[a-zA-Z0-9_\-\.\/]+", "<URL>", text)
    # 2. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

tfidf_settings = {
    "preprocessor": cybershield_text_preprocessor,
    "lowercase": True,
    "ngram_range": (1, 2),          # Unigrams + Bigrams (e.g. 'fuck you', 'shut up', 'you tool')
    "max_features": 40000,          # Optimal balance between expressiveness and RAM efficiency
    "min_df": 3,                    # Filters unique single-character noise and typos
    "max_df": 0.80,                 # Filters corpus-wide boilerplate without losing negation words
    "sublinear_tf": True,           # Applies 1 + log(tf) scaling to temper extreme word repetition
    "norm": "l2",                   # Euclidean unit vector normalization
    "token_pattern": r"(?u)\b\w+\b" # Preserves alphanumeric tokens and single-letter words
}
```

### Rationale for Settings:
- **`ngram_range=(1, 2)`:** Crucial in aggression detection. Single words like *"shut"* or *"tool"* are ambiguous, but bigrams like *"shut up"* or *"you tool"* carry unambiguous hostile polarity.
- **`sublinear_tf=True`:** Essential because angry users repeat aggressive words multiple times in a single comment (e.g., repeating insults 10 times). Log scaling ensures that 10 occurrences count for ~3.3x rather than 10x the weight of 1 occurrence.
- **`max_df=0.80`:** Avoids arbitrary external stop-word lists, which often inadvertently strip vital negation modifiers (`not`, `never`, `no`) that invert aggression polarity.

---

## 11. Next Steps (Pending User Approval)

With data preparation and architectural inspection complete, the following execution steps are ready to be triggered upon user approval:
1. **Script Creation:** Implement `src/train_text_model.py` with reproducible data splitting, preprocessor, and TF-IDF pipeline.
2. **Baseline Model Training:** Train and benchmark:
   - Logistic Regression (with balanced class weights)
   - Linear Support Vector Classifier (LinearSVC)
   - Multinomial Naive Bayes
3. **Evaluation Metrics:** Track Precision, Recall, F1-Score (Macro and Weighted), and PR-AUC.
4. **Save Model Artifacts:** Persist vectorizer and best-performing model to `models/`.

---

*Report prepared by Member 1 (AI & Social Network Analysis).*  
*No machine learning models have been trained. Awaiting user approval to proceed.*
