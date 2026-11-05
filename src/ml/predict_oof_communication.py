import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_data_prep import cybershield_url_normalizer, simple_norm

def generate_oof_communication_predictions(
    comm_path="data/processed/communication_cleaned_sorted.csv",
    clean_corpus_path="data/processed/text_corpus_clean_for_training.csv",
    output_path="data/processed/communication_with_oof_predictions.csv",
    n_splits=5,
    random_state=42
):
    """
    Generates strictly Out-Of-Fold (OOF) aggression predictions for the communication dataset
    with strict Duplicate-Group Isolation.

    Methodological Context & Leakage Prevention:
    In natural language corpora, identical or near-identical texts (differing only by whitespace,
    punctuation, or casing) can cross-contaminate folds if split randomly on raw strings.
    Furthermore, approximately 97.14% of the messages in the communication dataset overlap with
    the Wikipedia Detox text training corpus.

    This module enforces the following leakage-safe protocols:
    1. Normalization: Uses 'simple_norm' (lowercase, punctuation-stripped, space-collapsed) to
       construct deduplication equivalence classes (normalized message groups).
    2. Conflict Isolation: The 12 conflicting normalized texts (24 rows) identified during Stage 1
       are strictly excluded from the training pool of all folds.
    3. Stratified Group K-Fold: The clean pool of 128,201 unique normalized groups is partitioned
       across 5 folds using StratifiedGroupKFold. Every normalized message group belongs to
       EXACTLY ONE fold (0% normalized group overlap across training and held-out folds).
    4. Fold Training & Evaluation: For each fold, a model is trained on the other 4 folds.
       Messages belonging to fold k are scored strictly by the fold k model (which never saw
       that normalized string during training).
    5. Held-Out & Conflicting Messages: Unseen communication messages (2,585 events) and
       conflicting-label messages are scored via the 5-model ensemble average. We verify via
       explicit assertion that none of these messages ever entered the training set of any fold.
    6. Complete Coverage: All 90,356 communication events receive exactly one valid OOF probability.
    """
    print("=" * 75)
    print("CYBERSHIELD AI — DUPLICATE-SAFE OUT-OF-FOLD (OOF) STAGE 1 PREDICTOR")
    print("=" * 75)

    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load Datasets
    print(f"Loading communication data from: {comm_path}")
    df_comm = pd.read_csv(comm_path)
    total_comm = len(df_comm)
    print(f"  Total communication events: {total_comm:,}")

    print(f"Loading clean text corpus from: {clean_corpus_path}")
    df_clean = pd.read_csv(clean_corpus_path)
    total_clean = len(df_clean)
    print(f"  Total clean text corpus rows: {total_clean:,}")

    # 2. Text Normalization and Group Construction
    print("\n>>> Step 1: Constructing normalized message groups...")
    df_clean["norm_message"] = df_clean["message"].apply(simple_norm)
    df_comm["norm_message"] = df_comm["Message"].apply(simple_norm)

    # Identify and isolate 12 conflicting normalized strings (24 rows)
    conflicts = df_clean.groupby("norm_message")["label"].nunique()
    conf_norms = set(conflicts[conflicts > 1].index)
    n_conf_rows = int(df_clean["norm_message"].isin(conf_norms).sum())
    print(f"  Identified {len(conf_norms)} conflicting normalized groups ({n_conf_rows} rows).")
    print("  Excluding conflicting normalized groups from all supervised training folds.")

    df_clean_pool = df_clean[~df_clean["norm_message"].isin(conf_norms)].copy().reset_index(drop=True)
    n_unique_clean_groups = df_clean_pool["norm_message"].nunique()
    print(f"  Clean training pool: {len(df_clean_pool):,} rows across {n_unique_clean_groups:,} unique normalized groups.")

    # 3. Stratified Group K-Fold Partitioning
    print(f"\n>>> Step 2: Partitioning into {n_splits} duplicate-safe stratified folds...")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    group_to_fold = {}
    fold_train_groups = {}
    fold_val_groups = {}
    fold_models = {}

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df_clean_pool, df_clean_pool["label"], groups=df_clean_pool["norm_message"])):
        t_fold = time.time()
        fold_num = fold + 1

        tr_data = df_clean_pool.iloc[tr_idx]
        val_data = df_clean_pool.iloc[val_idx]

        tr_groups = set(tr_data["norm_message"])
        val_groups = set(val_data["norm_message"])

        # Strict group isolation check
        overlap = tr_groups.intersection(val_groups)
        assert len(overlap) == 0, f"CRITICAL LEAKAGE in fold {fold_num}: {len(overlap)} overlapping normalized groups!"

        fold_train_groups[fold_num] = tr_groups
        fold_val_groups[fold_num] = val_groups

        for g in val_groups:
            group_to_fold[g] = fold_num

        # Fit Stage 1 approved pipeline
        vec = TfidfVectorizer(
            preprocessor=cybershield_url_normalizer,
            lowercase=True,
            ngram_range=(1, 2),
            max_features=40000,
            min_df=3,
            max_df=0.80,
            sublinear_tf=True,
            norm="l2"
        )
        clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=random_state)
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        pipe.fit(tr_data["message"], tr_data["label"])

        fold_models[fold_num] = pipe
        print(f"  Fold {fold_num}/{n_splits}: {len(tr_data):,} train rows ({len(tr_groups):,} groups) | {len(val_data):,} val rows ({len(val_groups):,} groups) | Overlap: 0 (Trained in {time.time()-t_fold:.2f}s)")

    # Verify all clean groups assigned
    assert len(group_to_fold) == n_unique_clean_groups, "Mismatch in group fold assignments!"

    # 4. Generate Predictions for Communication Messages
    print("\n>>> Step 3: Generating duplicate-safe OOF predictions for 90,356 communication events...")
    
    # Categorize communication messages by source
    comm_groups = set(df_comm["norm_message"])
    clean_groups_set = set(group_to_fold.keys())

    overlapping_comm_groups = comm_groups.intersection(clean_groups_set)
    conflicting_comm_groups = comm_groups.intersection(conf_norms)
    unseen_comm_groups = comm_groups - clean_groups_set - conf_norms

    print(f"  Total unique normalized comm groups: {len(comm_groups):,}")
    print(f"  - Mapped to OOF Folds (1 to 5):        {len(overlapping_comm_groups):,} groups")
    print(f"  - Mapped to Conflicting Pool:         {len(conflicting_comm_groups):,} groups")
    print(f"  - Truly Unseen (Held-out from corpus): {len(unseen_comm_groups):,} groups")

    # Verify zero unseen groups ever entered any training set
    all_train_groups_union = set().union(*fold_train_groups.values())
    for g in unseen_comm_groups:
        assert g not in all_train_groups_union, f"Leakage detected: unseen group '{g}' was present in training data!"
    for g in conf_norms:
        assert g not in all_train_groups_union, f"Leakage detected: conflict group '{g}' was present in training data!"

    # Compute predictions per unique raw message for maximum computational efficiency
    unique_msgs_df = df_comm[["Message", "norm_message"]].drop_duplicates().reset_index(drop=True)
    print(f"\nScoring {len(unique_msgs_df):,} unique message texts across appropriate fold estimators...")

    msg_to_prob = {}
    msg_to_source = {}
    msg_to_status = {}

    # A. Unseen and conflicting messages: score with 5-model ensemble
    ensemble_msgs_df = unique_msgs_df[unique_msgs_df["norm_message"].isin(unseen_comm_groups | conflicting_comm_groups)]
    if len(ensemble_msgs_df) > 0:
        raw_texts = ensemble_msgs_df["Message"].tolist()
        ensemble_probs = np.mean([pipe.predict_proba(raw_texts)[:, 1] for pipe in fold_models.values()], axis=0)
        for text, norm_m, p in zip(raw_texts, ensemble_msgs_df["norm_message"], ensemble_probs):
            msg_to_prob[text] = float(p)
            if norm_m in conf_norms:
                msg_to_source[text] = "oof_conflict_ensemble_prediction"
                msg_to_status[text] = "conflict_ensemble_prediction"
            else:
                msg_to_source[text] = "oof_unseen_ensemble_prediction"
                msg_to_status[text] = "oof_ensemble_unseen_prediction"

    # B. Messages mapped to specific folds: score strictly with that fold's model
    for fold_num in range(1, n_splits + 1):
        fold_val_g = fold_val_groups[fold_num]
        fold_pipe = fold_models[fold_num]
        fold_train_g = fold_train_groups[fold_num]

        msgs_in_fold = unique_msgs_df[unique_msgs_df["norm_message"].isin(fold_val_g)]
        if len(msgs_in_fold) > 0:
            raw_texts = msgs_in_fold["Message"].tolist()
            fold_probs = fold_pipe.predict_proba(raw_texts)[:, 1]
            for text, norm_m, p in zip(raw_texts, msgs_in_fold["norm_message"], fold_probs):
                # Verify strictly out-of-sample
                assert norm_m not in fold_train_g, f"CRITICAL LEAKAGE: '{norm_m}' scored by fold {fold_num} model that trained on it!"
                msg_to_prob[text] = float(p)
                msg_to_source[text] = f"oof_fold_{fold_num}_prediction"
                msg_to_status[text] = "oof_cross_validation_prediction"

    # 5. Assemble and Validate Full Communication Output
    print("\n>>> Step 4: Assembling and validating final communication dataset...")
    df_out = df_comm.copy()
    
    df_out["oof_aggression_probability"] = df_out["Message"].map(msg_to_prob)
    df_out["aggression_probability"] = np.round(df_out["oof_aggression_probability"], 6)
    df_out["aggression_prediction"] = (df_out["aggression_probability"] >= 0.50).astype(int)
    df_out["prediction_status"] = df_out["Message"].map(msg_to_status)
    df_out["prediction_source"] = df_out["Message"].map(msg_to_source)

    # Verification assertions
    assert len(df_out) == total_comm, f"Record count mismatch: {len(df_out)} vs {total_comm}"
    assert df_out["oof_aggression_probability"].isnull().sum() == 0, "Null OOF probabilities detected!"
    assert df_out["aggression_probability"].isnull().sum() == 0, "Null aggression probabilities detected!"
    assert (df_out["oof_aggression_probability"] >= 0.0).all() and (df_out["oof_aggression_probability"] <= 1.0).all(), "OOF probabilities outside [0, 1]!"
    assert (df_out["aggression_probability"] >= 0.0).all() and (df_out["aggression_probability"] <= 1.0).all(), "Aggression probabilities outside [0, 1]!"

    # Save to output path
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved duplicate-safe OOF communication predictions: {output_path} ({len(df_out):,} rows)")

    # 6. Summary Statistics & Audit Output
    print("\n" + "=" * 50)
    print("OOF DUPLICATE LEAKAGE AUDIT SUMMARY:")
    print("=" * 50)
    print("Prediction Source Breakdown:")
    print(df_out["prediction_source"].value_counts())
    print("\nPrediction Status Breakdown:")
    print(df_out["prediction_status"].value_counts())
    print("\nAggression Classification Breakdown (threshold=0.50):")
    print(df_out["aggression_prediction"].value_counts())
    print(f"\nProbability Range: min={df_out['oof_aggression_probability'].min():.6f}, mean={df_out['oof_aggression_probability'].mean():.6f}, max={df_out['oof_aggression_probability'].max():.6f}")
    print(f"Total processing time: {time.time() - t0:.2f}s")
    print("=" * 75)

    return df_out

if __name__ == "__main__":
    generate_oof_communication_predictions()
