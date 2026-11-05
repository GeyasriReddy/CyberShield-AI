# CyberShield AI — Comprehensive Dataset Quality Audit Report
**Role:** Member 1 — Artificial Intelligence & Social Network Analysis  
**Date:** November 5, 2026  
**Audited Location:** `datasets/` (Mendeley Cyberbullying Benchmark Collection)  
**Output Reports:**
- [`results/data_quality_summary.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/data_quality_summary.csv)
- [`results/missing_values_report.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/missing_values_report.csv)
- [`results/duplicate_records_report.csv`](file:///c:/Users/vrind/Desktop/CyberShield_AI/results/duplicate_records_report.csv)

---

## 1. Executive Summary & Readiness Classification

A comprehensive data quality audit was conducted across all six CSV files in the Mendeley multi-modal cyberbullying dataset. The audit evaluated structural dimensions, data types, missing records, duplication, categorical and numerical distributions, text quality, temporal dynamics, and referential integrity across relational keys.

| Dataset Name | Rows | Columns | File Size | Missing Cells (%) | Exact Duplicates | Readiness Classification | Executive Recommendation |
|---|---|---|---|---|---|---|---|
| **[1. users_data.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/1.%20users_data.csv)** | 100 | 5 | 2.4 KB | 0.00% | 0 | **Ready** | Fully clean demographic baseline. No missing values or ID gaps. |
| **[2. peerness_values.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/2.%20peerness_values.csv)** | 4,950 | 3 | 108.7 KB | 0.00% | 0 | **Ready** | Complete undirected dyadic matrix ($\binom{100}{2} = 4,950$). Needs bidirectional expansion for directed queries. |
| **[3. Aggressive_All.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/3.%20Aggressive_All.csv)** | 118,828 | 2 | 31.3 MB | 0.01% (21) | 0 | **Requires investigation** | 68.55% text repetition (37,369 unique), 21 nulls, and 2,756 messages conflicting with non-aggressive corpus. |
| **[4. Non_Aggressive_All.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/4.%20Non_Aggressive_All.csv)** | 118,828 | 2 | 42.2 MB | 0.02% (50) | 0 | **Requires investigation** | 18.72% text repetition, 50 nulls, and 2,756 overlapping messages causing semantic label ambiguity. |
| **[5. Communication_Data_Among_Users.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/5.%20Communication_Data_Among_Users.csv)** | 90,356 | 6 | 31.1 MB | 0.00% | 0 | **Ready after cleaning** | Zero missing values; 100% unique messages. Requires temporal sorting per pair (only 12.36% monotonic) and dual-corpus resolution. |
| **[6. CB_Labels.csv](file:///c:/Users/vrind/Desktop/CyberShield_AI/datasets/6.%20CB_Labels.csv)** | 9,511 | 7 | 309.4 KB | 7.04% (4,686) | 0 | **Ready after cleaning** | 4,686 missing `Peerness` values (49.27% of column) due to asymmetric join; **100% recoverable** from `peerness_values.csv`. |

---

## 2. Dataset Architecture & Context

The CyberShield AI framework models cyberbullying as a multi-modal sociotechnical phenomenon:
1. **Node Demographics (`users_data.csv`):** 100 user nodes with age, gender, school, and grade.
2. **Social Power Dynamics (`peerness_values.csv`):** Peer status metric assessing social hierarchy parity between individuals.
3. **Lexical Aggression Corpora (`Aggressive_All.csv` & `Non_Aggressive_All.csv`):** Benchmark NLP message pools derived from online discussion platforms.
4. **Dyadic Interaction Network (`Communication_Data_Among_Users.csv`):** Longitudinal directed messaging events among the 100 users across 14 months (Jan 2021 – Feb 2022).
5. **Ground Truth Dyadic Cyberbullying Labels (`CB_Labels.csv`):** Dyad-level classifications capturing repeated aggression, intent to harm, and power imbalance.

---

## 3. Comprehensive File-by-File Audit

### 3.1 `1. users_data.csv`
- **Dimensions:** 100 rows, 5 columns.
- **Columns & Data Types:**
  - `UserID`: `int64` (Continuous integer primary key from 1 to 100; 0 missing, 0 duplicates).
  - `Age`: `int64` (Range: 8 to 18 years; Mean: 12.66, Median: 13.0, Std: 3.21).
  - `Gender`: `object` (`Others`: 44, `Female`: 30, `Male`: 26; 0 nulls, 0 whitespace).
  - `School Name`: `object` (15 distinct schools: `School1` to `School15`; max 12 users in `School1`/`School15`, min 1 user in `School6`).
  - `Grade`: `int64` (Range: Grade 2 to Grade 12; Mean: 6.66, Median: 7.0).
- **Domain Consistency Check:**
  - An exact deterministic linear relationship exists: $\text{Grade} = \text{Age} - 6$ for all 100 users.
  - No age/grade outliers (0 records outside $1.5 \times \text{IQR}$).
- **Referential Integrity:** Acts as the ground-truth user registry. All 100 UserIDs are accounted for without gaps.

---

### 3.2 `2. peerness_values.csv`
- **Dimensions:** 4,950 rows, 3 columns.
- **Columns & Data Types:**
  - `User 1`: `int64` (Range: 1 to 99).
  - `User 2`: `int64` (Range: 2 to 100).
  - `Peerness`: `float64` (Mean: 0.5359, Std: 0.1854, Min: 0.1000, Max: 0.9967).
- **Structural Properties:**
  - Row count matches the theoretical number of undirected dyads for 100 nodes:
    $$\binom{100}{2} = \frac{100 \times 99}{2} = 4,950$$
  - In 100% of rows, `User 1 < User 2`. There are no self-loops (`User 1 == User 2`: 0) and zero duplicate pairs.
  - Documented range $[0, 1]$ is fully respected; zero statistical outliers detected.
- **Referential Integrity:** 100% of `User 1` and `User 2` IDs exist in `users_data.csv`.

---

### 3.3 `3. Aggressive_All.csv`
- **Dimensions:** 118,828 rows, 2 columns.
- **Columns & Data Types:**
  - `No.`: `int64` (Sequential index 1 to 118,828; strictly unique).
  - `Message`: `object` (Free text).
- **Missing & Empty Records:**
  - **21 records contain NaN/Null messages** (e.g., indices 9152, 15233, 16240, 27094, 30329).
  - 0 empty strings `""` or whitespace-only records among non-null rows.
- **Duplication & Content Quality:**
  - **High Duplication:** Only 37,369 distinct messages exist across the 118,828 rows (**81,458 duplicate messages, 68.55% duplication rate**). Highly aggressive phrases, insults, and profanity recur hundreds of times.
  - Character length: Min 2, Max 10,376, Mean 268.4, Median 109.0.
  - Word count: Min 1, Max 2,832, Mean 48.7, Median 20.0.
  - Extreme short messages: 198 messages contain $\le 1$ word; 87 messages contain $< 5$ characters (e.g., `"Baby"`, `"NAME"`, `"TRUE"`, `"You tool"`).
  - Anonymization and preprocessing artifacts: Contains 189 occurrences of the `NAME` mask token and 1,372 occurrences of the `IP` mask token.
  - Stripped URL artifacts: 3,560 messages contain URLs with protocol punctuation removed (e.g., `httpmongomutterblogspotcomonedownhtml`).
  - Casing: 6,682 messages written in ALL CAPS.
  - Non-ASCII: 119 messages contain non-ASCII characters.

---

### 3.4 `4. Non_Aggressive_All.csv`
- **Dimensions:** 118,828 rows, 2 columns.
- **Columns & Data Types:**
  - `No.`: `int64` (Sequential index 1 to 118,828; strictly unique).
  - `Message`: `object` (Free text).
- **Missing & Empty Records:**
  - **50 records contain NaN/Null messages** (e.g., indices 1223, 4525, 7111, 9166, 14511).
  - 0 empty strings `""` or whitespace-only records among non-null rows.
- **Duplication & Content Quality:**
  - **Moderate Duplication:** 96,581 distinct messages exist across 118,828 rows (**22,246 duplicate messages, 18.72% duplication rate**).
  - Character length: Min 1, Max 14,397, Mean 364.1, Median 172.0.
  - Word count: Min 1, Max 2,500, Mean 63.7, Median 31.0.
  - Extreme short messages: 314 messages contain $\le 1$ word; 109 messages contain $< 5$ characters.
  - Anonymization artifacts: 53 occurrences of `NAME` token; 1,841 occurrences of `IP` token.
  - Stripped URL artifacts: 5,175 messages contain stripped URLs.
  - Casing: 672 messages in ALL CAPS.
  - Non-ASCII: 604 messages contain non-ASCII characters.

---

### 3.5 `5. Communication_Data_Among_Users.csv`
- **Dimensions:** 90,356 rows, 6 columns.
- **Columns & Data Types:**
  - `Date`: `object` (Format `YYYY-MM-DD`, range: `2021-01-01` to `2022-02-28`; 0 nulls, 0 invalid dates).
  - `Time`: `object` (Format `HH:MM`, 24-hour; 0 invalid times).
  - `User1 ID`: `int64` (Sender ID; 100% valid within range 1 to 100).
  - `User2 ID`: `int64` (Receiver ID; 100% valid within range 1 to 100).
  - `Message`: `object` (Free text; 0 nulls, 0 empty strings, **all 90,356 messages are distinct text strings**).
  - `Label`: `int64` (Message-level aggression indicator: `0` = 59,579 [65.94%], `1` = 30,777 [34.06%]).
- **Network & Interaction Topology:**
  - 9,511 directed sender-receiver pairs $(User1 \to User2)$.
  - Zero self-loops (`User1 ID == User2 ID`: 0).
  - Zero full-row duplicates across `(Date, Time, User1 ID, User2 ID, Message)`.
- **Temporal Order Finding:**
  - Only **1,176 pairs (12.36%)** have messages ordered chronologically in the file.
  - In **8,335 pairs (87.64%)**, timestamps fluctuate out of sequence, necessitating a chronological sort prior to dynamic graph or sequential RNN/LSTM modeling.

---

### 3.6 `6. CB_Labels.csv`
- **Dimensions:** 9,511 rows, 7 columns.
- **Columns & Data Types:**
  - `User1 ID`: `int64` (Directed sender / aggressor candidate).
  - `User2 ID`: `int64` (Directed receiver / target candidate).
  - `Total_messages`: `int64` (Range: 1 to 40; Mean: 9.50, Median: 8.0).
  - `Aggressive_Count`: `int64` (Range: 0 to 27; Mean: 3.24, Median: 2.0).
  - `Intent_to_Harm`: `float64` (Range: 0.0125 to 0.8337; Mean: 0.2702, Median: 0.2750).
  - `Peerness`: `float64` (**4,686 nulls [49.27%]**, 4,825 non-null; Range: 0.1000 to 0.9967).
  - `CB_Label`: `int64` (`0`: 8,519 [89.57%], `1`: 992 [10.43%]).
- **Exact Aggregation Consistency:**
  - When aggregating `Communication_Data_Among_Users.csv` by `(User1 ID, User2 ID)`:
    - Directed dyad count: Exactly **9,511 pairs** (100.00% match).
    - Message counts: `Total_messages` matches the grouped message count for **all 9,511 rows** (0 mismatches).
    - Aggression counts: `Aggressive_Count` matches the sum of `Label == 1` for **all 9,511 rows** (0 mismatches).
- **Target Variable Imbalance:**
  - Significant class imbalance: 8,519 non-cyberbullying dyads vs 992 cyberbullying dyads (~8.6 : 1 ratio).

---

## 4. Special Deep-Dive Findings

### 4.1 Root Cause & Mathematical Resolution of Missing Peerness in `CB_Labels.csv`
One of the most consequential discoveries of this audit is why `CB_Labels.csv` has **4,686 missing values in the `Peerness` column (49.27%)**:

1. **Root Cause Analysis:**
   - In `peerness_values.csv`, peerness is represented as an undirected graph where only canonical pairs $User 1 < User 2$ are stored (4,950 rows).
   - In `CB_Labels.csv`, records represent **directed** communication dyads $(User1 \to User2)$ across 9,511 pairs.
   - When the dataset creators generated `CB_Labels.csv`, they executed a non-symmetric left join:
     $$\text{ON } CB\_Labels.User1 == peerness.User1 \quad \text{AND} \quad CB\_Labels.User2 == peerness.User2$$
   - Consequently:
     - For every row where $User1 < User2$ (4,825 rows), `Peerness` was successfully populated.
     - For every row where $User1 > User2$ (**4,686 rows**), the join produced a `NULL` (`NaN`).

2. **Verification & Proof of 100% Recovery:**
   - By constructing a symmetric lookup dictionary from `peerness_values.csv`:
     $$\text{Peerness}(u_1, u_2) = \text{Peerness}(u_2, u_1)$$
   - Exactly **4,686 of the 4,686 missing values (100.00%)** were matched and validated.
   - **Remedy:** No records should be dropped. The 4,686 missing values can be completely and losslessly imputed via symmetric lookup during preprocessing.

```
+-----------------------------------------------------------------------------------------+
|                                    PEERNESS JOIN ANOMALY                                |
|                                                                                         |
|  peerness_values.csv (4,950 pairs)            CB_Labels.csv (9,511 directed dyads)      |
|  +--------+--------+----------+               +----------+----------+----------+        |
|  | User 1 | User 2 | Peerness |               | User1 ID | User2 ID | Peerness |        |
|  +--------+--------+----------+               +----------+----------+----------+        |
|  |   1    |   2    |  0.500   | ------------> |    1     |    2     |  0.500   | (OK)   |
|  |  ...   |  ...   |   ...    |               |    2     |    1     |   NaN    | (BUG!) |
|  +--------+--------+----------+               +----------+----------+----------+        |
|                                                     ^                                   |
|                                                     | Symmetrize (u2, u1)               |
|                                                     +--- 100% Lossless Recovery!        |
+-----------------------------------------------------------------------------------------+
```

---

### 4.2 Text Quality & Cross-Corpus Contamination (2,756 Overlapping Messages)
The audit identified cross-corpus contamination between `Aggressive_All.csv` and `Non_Aggressive_All.csv`:

1. **Overlap Extent:**
   - There are **2,756 distinct messages** that appear in **both** `Aggressive_All.csv` and `Non_Aggressive_All.csv`.
2. **Impact on Communication Data:**
   - In `Communication_Data_Among_Users.csv`, exactly **2,596 messages** are from this overlapping set.
   - Of these 2,596 messages:
     - **838 messages** are labeled as non-aggressive (`Label = 0`).
     - **1,758 messages** are labeled as aggressive (`Label = 1`).
3. **Semantic Analysis of Conflicting Texts:**
   - Many overlapping messages contain sarcastic remarks, administrative warnings, or contentious debate language (e.g., talk-page disputes on Wikipedia).
   - *Example Overlap 1:* `"dont try to threaten me with barring i dont care at all..."` (Aggressive in one context, administrative pushback in another).
   - *Example Overlap 2:* `"wtf what is your problem Im not spamming just adding a relative link..."`
4. **Machine Learning Implication:**
   - Training a standard NLP classifier directly on the merged corpus without deduplication will inject conflicting supervisory signals (identical features mapped to opposite binary labels).

---

### 4.3 Empirical Proof: Aggression Does Not Equal Cyberbullying
A foundational principle of social science and anti-bullying research is that **cyberbullying is not merely the occurrence of aggressive language**, but requires:
1. Intent to inflict emotional or psychological harm.
2. Repetition over time.
3. Power asymmetry between perpetrator and victim.

The audited data provides empirical proof of this principle:

| Metric | Non-Cyberbullying Dyads (`CB_Label = 0`) | Cyberbullying Dyads (`CB_Label = 1`) |
|---|---|---|
| **Total Dyad Count** | 8,519 (89.57%) | 992 (10.43%) |
| **Mean Total Messages** | 8.11 | 21.44 |
| **Mean Aggressive Messages** | 2.43 | 10.14 |
| **Min Aggressive Messages** | 0 | **5** |
| **Max Aggressive Messages** | **22** | 27 |
| **Mean Intent to Harm** | 0.2419 | 0.5136 |
| **Min Intent to Harm** | 0.0125 | **0.4042** |
| **Max Intent to Harm** | **0.7567** | 0.8337 |
| **Mean Peerness** | 0.5265 | 0.6135 |

**Key Analytical Insights:**
- There are user dyads with as many as **22 aggressive messages** that are explicitly labeled **non-cyberbullying (`CB_Label = 0`)** because the interaction represents symmetric mutual conflict or playful banter among close peers rather than systematic bullying.
- Conversely, dyads with as few as **5 aggressive messages** are classified as **cyberbullying (`CB_Label = 1`)** when concentrated with high `Intent_to_Harm` and asymmetric status.
- This confirms that Member 1 must engineer multi-modal graph and interaction features (social hierarchy, centrality, reciprocity, temporal bursts) rather than relying exclusively on NLP text toxicity scores.

---

### 4.4 Communication Temporal Dynamics & Graph Topology
- **Temporal Distribution:**
  - 2021: 77,737 messages (86.03%).
  - 2022 (Jan-Feb): 12,619 messages (13.97%).
  - Peak interaction dates occur around school term sessions.
- **Sequential Disorder:**
  - Out of 9,511 communicating pairs, **only 1,176 pairs (12.36%)** have strictly chronological message timestamps.
  - In 8,335 pairs, chronological sequences are mixed across different dates.
  - To compute dynamic social network metrics (e.g., dynamic in-degree centrality, burstiness, inter-arrival times), messages must be sorted by `['User1 ID', 'User2 ID', 'Date', 'Time']`.

---

## 5. Referential Integrity & Relational Verification Matrix

| Primary Key / Foreign Key Relationship | Source Column | Target Column | Integrity Result | Notes |
|---|---|---|---|---|
| `users_data.UserID` $\to$ Primary Key | `UserID` | N/A | **Valid (100/100)** | Monotonic 1..100, zero gaps, zero duplicates. |
| `peerness_values` $\to$ `users_data` | `User 1`, `User 2` | `users_data.UserID` | **Valid (4,950/4,950)** | All endpoints belong to [1, 100]. |
| `Communication_Data` $\to$ `users_data` | `User1 ID`, `User2 ID` | `users_data.UserID` | **Valid (90,356/90,356)** | Zero orphan senders or receivers. |
| `CB_Labels` $\to$ `users_data` | `User1 ID`, `User2 ID` | `users_data.UserID` | **Valid (9,511/9,511)** | All dyad endpoints belong to [1, 100]. |
| `CB_Labels` $\leftrightarrow$ `Communication_Data` | `(User1 ID, User2 ID)` | `(User1 ID, User2 ID)` | **Exact Match (9,511/9,511)** | Bijective mapping; zero dangling pairs. |
| Message Count Validation | `CB_Labels.Total_messages` | `count(Comm.Message)` | **Exact Match (9,511/9,511)** | $\Delta = 0$ for all records. |
| Aggression Count Validation | `CB_Labels.Aggressive_Count` | `sum(Comm.Label)` | **Exact Match (9,511/9,511)** | $\Delta = 0$ for all records. |
| Peerness Direct Recovery | `CB_Labels.Peerness` (nulls) | `peerness_values(u2, u1)` | **100% Recoverable (4,686/4,686)** | Lossless symmetric recovery. |

---

## 6. Actionable Preprocessing & Cleaning Roadmap

Before training machine learning models or constructing the social network graph, Member 1 should execute the following non-destructive pipeline:

### Step 1: Lossless Recovery of Peerness in Dyad Table
- Do not delete any rows in `CB_Labels.csv`.
- Create a symmetrized lookup dictionary from `peerness_values.csv`:
  $$\mathcal{P}(u_i, u_j) = \mathcal{P}(u_j, u_i)$$
- Fill all 4,686 `NaN` entries in `CB_Labels.csv` using this symmetric lookup. Output to `data/processed/cb_dyads_cleaned.csv`.

### Step 2: Temporal Ordering of Communication Graph
- Parse `Date` and `Time` into a unified ISO-8601 `timestamp` (`pd.to_datetime(df['Date'] + ' ' + df['Time'])`).
- Sort `Communication_Data_Among_Users.csv` chronologically by `['User1 ID', 'User2 ID', 'timestamp']`.
- Output to `data/processed/communication_sorted.csv`.

### Step 3: Text Normalization & NLP Corpus Deduplication
- For NLP training sets (`Aggressive_All` and `Non_Aggressive_All`):
  - Remove the 21 null aggressive messages and 50 null non-aggressive messages.
  - Flag the 2,756 overlapping messages to prevent conflicting ground-truth in supervised loss functions.
  - Strip redundant leading/trailing whitespace, fix non-ASCII artifacts, and standardize tokens (`NAME`, `IP`).
  - Perform deduplication when evaluating lexical models to avoid severe test-set leakage.

### Step 4: Class Imbalance Strategy for Cyberbullying Modeling
- Cyberbullying is present in only 10.43% of dyads.
- Member 1 should implement stratified train/validation/test splits grouped by user or dyad.
- Employ cost-sensitive loss functions, focal loss, or PR-AUC / F1-Macro evaluation metrics rather than raw accuracy.

---

*Report compiled by Member 1 (AI & Social Network Analysis).*  
*Audited with Python 3.14 on verified local datasets.*
