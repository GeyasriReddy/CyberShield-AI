# CyberShield AI — Machine Learning Results & Evaluation Report
**Component:** Stage 1 — Aggressive vs. Non-Aggressive Language Classification  
**Member:** Member 1 (Artificial Intelligence & Social Network Analysis)  
**Date:** November 5, 2026  
**Pipeline Code:**
- [`src/ml/train_model.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml/train_model.py)
- [`src/ml/calibrate.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml/calibrate.py)
- [`src/ml/evaluate.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml/evaluate.py)
- [`src/ml/predict.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/src/ml/predict.py)
**Saved Models:**
- [`models/best_model_pipeline.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/best_model_pipeline.joblib) (End-to-End Inference Pipeline)
- [`models/tfidf_vectorizer.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/tfidf_vectorizer.joblib) (Vocabulary size: 40,000)
- [`models/calibrated_logistic.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/calibrated_logistic.joblib)
- [`models/calibrated_svm.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/calibrated_svm.joblib)
- [`models/logistic_regression_baseline.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/logistic_regression_baseline.joblib)
- [`models/linear_svm_baseline.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/linear_svm_baseline.joblib)
- [`models/frequency_weighted_model.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/frequency_weighted_model.joblib)
**Generated Visualizations & Data:**
- [`results/model_comparison.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/model_comparison.csv)
- [`results/confusion_matrix_logistic.png`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/confusion_matrix_logistic.png)
- [`results/confusion_matrix_svm.png`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/confusion_matrix_svm.png)
- [`results/precision_recall_curve.png`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/precision_recall_curve.png)
- [`results/calibration_curves.png`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/calibration_curves.png)

---

## 1. Executive Summary

The machine learning text classification component for CyberShield AI has been built, trained, calibrated, and rigorously evaluated. This model serves as the **Stage 1 message aggression classifier**, predicting:
- **`1` = Aggressive Language** (toxic, hostile, abusive, offensive, or insulting text)
- **`0` = Non-Aggressive Language** (constructive, neutral, civil, or polite text)

### Critical Distinction Preserved
This model is an **aggression detector**, not a cyberbullying detector. A hostile message is not automatically cyberbullying; cyberbullying requires dyadic repetition over time, social hierarchy power disparity (peerness), and intent to harm. The calibrated aggression probabilities from this model will serve as inputs to the downstream Stage 2 Social Network Analysis (SNA) graph classifier.

---

## 2. Experimental Setup & Leakage Safeguards

### 2.1 Dataset Partitions
All models were trained and tuned strictly on the pre-partitioned, leakage-free datasets:
- **Training Set (`ml_corpus_train_split.csv`):** 89,885 messages (24,218 aggressive [26.94%], 65,667 non-aggressive [73.06%])
- **Validation Set (`ml_corpus_val_split.csv`):** 19,258 messages (5,190 aggressive [26.95%], 14,068 non-aggressive [73.05%])
- **Test Set (`ml_corpus_test_split.csv`):** 19,271 messages (5,193 aggressive [26.95%], 14,078 non-aggressive [73.05%])

### 2.2 Leakage Guardrails Enforced
1. **Zero Normalization Overlap:** Equivalent normalized messages (`norm_message`) were grouped together so that capitalization/whitespace variants never cross split boundaries.
2. **Train-Only Vectorizer Fitting:** The `TfidfVectorizer` was fitted **exclusively on the 89,885 training records**. The validation and test sets were transformed using the frozen vocabulary.
3. **No Test Data in Tuning or Calibration:** Hyperparameters ($C$) and Platt probability calibration were optimized strictly using the validation set and 3-fold cross-validation on training data. The test dataset was touched **exactly once** for final reporting.

---

## 3. TF-IDF Vectorization Architecture

The TF-IDF feature extractor was initialized with the approved domain configuration:
- `preprocessor`: `cybershield_url_normalizer` (maps standard and punctuation-stripped URLs to `<url>`).
- `lowercase`: `True`
- `ngram_range`: `(1, 2)` (captures vital unigrams and bigrams like `"shut up"`, `"fuck you"`, `"you tool"`, `"go away"`).
- `max_features`: `40,000`
- `min_df`: `3` (drops single-instance noise and typos).
- `max_df`: `0.80` (filters boilerplate without dropping negation words like `not` and `no`).
- `sublinear_tf`: `True` ($tf = 1 + \log(tf)$ prevents extreme repetition from disproportionately skewing weights).
- `norm`: `'l2'`

**Execution Metric:** Fitted in 21.32 seconds. Vocabulary density: Exactly 40,000 features. Feature matrices: Train $(89885, 40000)$, Val $(19258, 40000)$, Test $(19271, 40000)$.

---

## 4. Model Tuning & Hyperparameter Selection

Hyperparameter exploration on the validation split evaluated Macro F1, Aggressive F1, and Recall:

### 4.1 Logistic Regression Tuning (`class_weight='balanced'`)
| Regularization $C$ | Validation Accuracy | Macro F1-Score | Aggressive F1-Score | Aggressive Recall |
|---|---|---|---|---|
| $C = 0.1$ | 84.55% | 81.68% | 74.43% | 83.45% |
| $C = 1.0$ | 87.54% | 84.83% | 78.43% | **84.05%** |
| **$C = 5.0$ (Selected)** | **87.94%** | **85.20%** | **78.83%** | 83.29% |
| $C = 10.0$ | 87.74% | 84.89% | 78.33% | 82.22% |

*Decision:* $C = 5.0$ achieved the highest Validation Macro F1 (85.20%) and Aggressive F1 (78.83%).

### 4.2 Linear Support Vector Classifier Tuning (`class_weight='balanced'`)
| Regularization $C$ | Validation Accuracy | Macro F1-Score | Aggressive F1-Score | Aggressive Recall |
|---|---|---|---|---|
| $C = 0.01$ | 84.63% | 81.76% | 74.53% | 83.43% |
| **$C = 0.1$ (Selected)** | **87.76%** | **85.07%** | **78.73%** | **84.08%** |
| $C = 1.0$ | 87.38% | 84.47% | 77.75% | 81.81% |
| $C = 5.0$ | 85.56% | 82.19% | 74.43% | 78.00% |

*Decision:* $C = 0.1$ achieved the highest Validation Macro F1 (85.07%) and Aggressive Recall (84.08%).

---

## 5. Confidence Calibration (Platt Scaling)

Raw SVM decision scores are signed geometric distances to the hyperplane ($w^T x + b$), **not probabilities**. Similarly, uncalibrated logistic regression under class imbalance can produce overconfident probabilities.

### 5.1 Methodology
- Implemented **Platt scaling** via `CalibratedClassifierCV(cv=3, method='sigmoid')` fitted on the training split.
- Evaluated calibration quality using the **Brier score** (mean squared difference between predicted probability and actual binary label, where $0.0$ is perfect calibration):
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (P(\hat{y}_i = 1) - y_i)^2$$

### 5.2 Calibration Results & Brier Score Comparison
- **Uncalibrated Logistic Regression:** Brier Score = `0.0888`
- **Calibrated Logistic Regression:** Brier Score = **`0.0818` (8% calibration improvement!)**
- **Calibrated Linear SVM:** Brier Score = **`0.0841`**
- Both calibrated models show tight alignment with the ideal diagonal on the calibration reliability curves ([results/calibration_curves.png](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/calibration_curves.png)).

### 5.3 Meaning & Practical Limitations of Confidence Scores
- **Meaning:** A calibrated confidence score of $0.85$ means that among comments receiving this score, approximately $85\%$ are genuinely aggressive.
- **Limitations:**
  1. **Lexical Artifacts:** Confidence reflects vocabulary frequency; highly toxic slurs yield $\approx 0.99$, while passive-aggressive sarcasm without overt profanity may receive lower confidence ($0.45 - 0.60$).
  2. **Short Text Sensitivity:** Single-word insults or ambiguous acronyms have high variance.
  3. **Domain Transfer:** Confidence scores assume text resembles Wikipedia talk discourse and may shift on youth slang.

---

## 6. Frequency-Weighted Experiment (Task 6)

We conducted a separate experiment incorporating preserved corpus occurrence frequencies without data leakage:
$$w_i = 1.0 + \log(1.0 + \text{frequency}_i)$$
Weights were derived **strictly from `ml_corpus_train_split.csv`** (mean weight: 1.93, max: 4.83).

### Comparison: Baseline vs. Frequency-Weighted Model
- **Test Accuracy:** **88.76%** (highest overall accuracy across all models).
- **Test Precision:** **81.60%** (vs 75.09% in uncalibrated baseline).
- **Test Recall:** **75.26%** (catches 3,908 aggressive messages, significantly higher recall than calibrated baseline's 72.23%).
- **Macro F1:** **85.36%**.
- **Insight:** Weighting by log-frequency rewarded common conversational insults without letting single viral spam phrases overwhelm loss gradients.

---

## 7. Final Test Evaluation Benchmark (Untouched Test Split)

The final evaluation was conducted on the **19,271 untouched test records** ([`results/model_comparison.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/model_comparison.csv)):

| Model | Test Accuracy | Precision (Agg) | Recall (Agg) | F1 (Agg) | Macro F1 | PR-AUC | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | 88.18% | 75.09% | **84.04%** | **79.31%** | **85.52%** | 86.74% | 94.09% | 0.0888 |
| **Calibrated Logistic Regression** | 88.73% | **83.71%** | 72.23% | 77.55% | 85.01% | **86.91%** | **94.26%** | **0.0818** |
| **Linear SVM (Balanced)** | 87.96% | 74.37% | 84.38% | 79.06% | 85.30% | *N/A* | *N/A* | *N/A* |
| **Calibrated Linear SVM** | 88.37% | 82.37% | 72.33% | 77.02% | 84.62% | 86.09% | 94.00% | 0.0841 |
| **Frequency-Weighted Logistic Regression** | **88.76%** | 81.60% | 75.26% | 78.30% | 85.36% | 86.79% | 94.03% | 0.0831 |

### Confusion Matrix Breakdown (Test Set: $N = 19,271$)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | False Positive Rate |
|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | 12,630 | 1,448 | **829** | **4,364** | 10.29% |
| **Calibrated Logistic Regression** | **13,348** | **730** | 1,442 | 3,751 | **5.18% (Halved!)** |
| **Linear SVM (Balanced)** | 12,568 | 1,510 | **811** | **4,382** | 10.73% |
| **Calibrated Linear SVM** | 13,274 | 804 | 1,437 | 3,756 | 5.71% |
| **Frequency-Weighted Logistic Regression** | 13,197 | 881 | 1,285 | 3,908 | 6.26% |

---

## 8. Analytical Findings & Recommendations

### 8.1 Which Model Performed Better?
- **For High Recall / Screening (Flagging Potential Harm):**
  - **Balanced Logistic Regression ($C=5.0$)** achieved the highest aggressive recall (**84.04%**, catching 4,364 / 5,193 toxic comments) and highest Macro F1 (**85.52%**).
- **For High Precision / Low False Alarms (Automated Action):**
  - **Calibrated Logistic Regression** cut false positives by nearly 50% (from 1,448 down to 730), raising precision to **83.71%** with the best PR-AUC (**86.91%**) and best calibration (Brier score **0.0818**).
- **For Balanced Real-World Performance:**
  - **Frequency-Weighted Logistic Regression** struck the best compromise: **88.76% accuracy**, **81.60% precision**, and **75.26% recall**.

### 8.2 Which Model Should Be Saved for Deployment?
We selected and saved **Calibrated Logistic Regression** as the primary end-to-end pipeline:
[`models/best_model_pipeline.joblib`](file:///c:/Users/vrind/Desktop/CyberShield_AI/models/best_model_pipeline.joblib)

**Rationale:**
1. **Calibrated Probabilities for Downstream SNA:** In Stage 2, Member 1 will feed continuous aggression probabilities $P(\text{aggression} \mid m_t) \in [0, 1]$ into dyadic graph edge features. Having a Brier score of **0.0818** guarantees that these probabilities are mathematically sound.
2. **Tunable Decision Thresholds:** With calibrated probabilities, the operating threshold can be adjusted dynamically (e.g. $\tau = 0.35$ for high-recall auditing, or $\tau = 0.70$ for automated warnings).

### 8.3 Is Confidence Calibration Reliable?
**Yes.** As visualized in [results/calibration_curves.png](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/calibration_curves.png), Platt scaling with 3-fold cross-validation brought the reliability curve into close conformity with the diagonal, reducing the Brier score to `0.0818`.

### 8.4 Remaining Problems & Considerations for Next Steps
1. **Context-Dependent Sarcasm:** Subtle sarcasm without overt aggression is occasionally missed (e.g. false negatives). This is expected for pure n-gram bag-of-words models and reinforces why Stage 2 (Social Network Analysis: user repetition, dyad peerness, interaction graphs) is essential.
2. **URL Tokens:** The `<url>` placeholder effectively neutralizes stripped URL artifacts, preventing out-of-vocabulary crashes.

---

## 9. Automated Test Suite Verification

All **38 unit tests** across seven test modules passed in 42.70 seconds:

```text
Ran 38 tests in 42.700s

OK
```

The new test suite [`tests/test_ml_training_and_inference.py`](file:///c:/Users/vrind/Desktop/CyberShield_AI/tests/test_ml_training_and_inference.py) verified:
1. `test_model_artifact_loading`: All 7 joblib model files load cleanly from disk.
2. `test_new_comment_prediction`: Correctly predicts aggressive vs polite comments.
3. `test_confidence_generation`: Probabilities sum to 1.0 and match confidence bounds.
4. `test_label_mapping`: Verified `1` $\to$ Aggressive and `0` $\to$ Non-Aggressive.
5. `test_reproducibility`: Deterministic predictions across multiple invocations.
6. `test_no_train_test_leakage_in_vectorizer`: Vocabulary fits strictly on train split with sublinear TF.
7. `test_evaluation_calculations_consistency`: Mathematical consistency of all confusion matrix derivations.

---

*Report prepared by Member 1 (AI & Social Network Analysis).*  
*Stage 1 text classification complete. Ready to proceed to Social Network Analysis & Graph Feature Engineering upon user request.*
