import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Compiled regex for genuine URLs (both standard and punctuation-stripped)
URL_REGEX = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b(?:https?|www)[a-z0-9_\-\.\/]{4,}\b"
)

def cybershield_url_normalizer(text, url_token="<url>"):
    """
    Normalizes text for ML classification:
    1. Converts text to lowercase.
    2. Replaces genuine raw and punctuation-stripped URLs with url_token.
    3. Preserves all ordinary words, punctuation, and vital negations.
    4. Normalizes extraneous whitespace.
    """
    text = str(text).lower()
    text = URL_REGEX.sub(url_token, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_norm(text):
    """
    Standard normalization for deduplication grouping:
    Lowercases, strips punctuation, and collapses spaces.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()

def prepare_ml_splits():
    print("=" * 70)
    print("CYBERSHIELD AI — MEMBER 1: ML DATA SPLITTING & PREPARATION")
    print("=" * 70)

    f_train = "data/processed/text_corpus_clean_for_training.csv"
    processed_dir = "data/processed"
    results_dir = "results"

    df = pd.read_csv(f_train)
    df["norm_message"] = df["message"].apply(simple_norm)

    # 1. Identify and exclude 12 conflicting normalized texts (24 rows)
    conflicts = df.groupby("norm_message")["label"].nunique()
    conf_norms = set(conflicts[conflicts > 1].index)
    
    df_conflicts = df[df["norm_message"].isin(conf_norms)].copy().reset_index(drop=True)
    df_clean_pool = df[~df["norm_message"].isin(conf_norms)].copy().reset_index(drop=True)

    print(f"Total raw training pool: {len(df)}")
    print(f"Excluded conflicting rows: {len(df_conflicts)} (12 normalized texts)")
    print(f"Clean pool for splitting: {len(df_clean_pool)}")

    # Save excluded conflicts file without overwriting existing files
    conflict_path = os.path.join(processed_dir, "ml_corpus_conflict_excluded.csv")
    df_conflicts.to_csv(conflict_path, index=False)
    print(f"Saved: {conflict_path} ({len(df_conflicts)} rows)")

    # 2. Stratified Group Splitting (70% Train, 15% Val, 15% Test)
    group_df = df_clean_pool.groupby("norm_message").agg(
        label=("label", "first"),
        row_count=("message", "count")
    ).reset_index()

    # Split 1: 70% train, 30% temp
    groups_train, groups_temp = train_test_split(
        group_df,
        test_size=0.30,
        random_state=42,
        stratify=group_df["label"]
    )

    # Split 2: 15% val, 15% test
    groups_val, groups_test = train_test_split(
        groups_temp,
        test_size=0.50,
        random_state=42,
        stratify=groups_temp["label"]
    )

    train_groups = set(groups_train["norm_message"])
    val_groups = set(groups_val["norm_message"])
    test_groups = set(groups_test["norm_message"])

    df_train_split = df_clean_pool[df_clean_pool["norm_message"].isin(train_groups)].copy().reset_index(drop=True)
    df_val_split = df_clean_pool[df_clean_pool["norm_message"].isin(val_groups)].copy().reset_index(drop=True)
    df_test_split = df_clean_pool[df_clean_pool["norm_message"].isin(test_groups)].copy().reset_index(drop=True)

    # Verify zero normalized overlap
    s_train = set(df_train_split["norm_message"])
    s_val = set(df_val_split["norm_message"])
    s_test = set(df_test_split["norm_message"])

    assert len(s_train.intersection(s_val)) == 0, "Train-Val overlap detected!"
    assert len(s_train.intersection(s_test)) == 0, "Train-Test overlap detected!"
    assert len(s_val.intersection(s_test)) == 0, "Val-Test overlap detected!"

    # Save split datasets
    p_train = os.path.join(processed_dir, "ml_corpus_train_split.csv")
    p_val = os.path.join(processed_dir, "ml_corpus_val_split.csv")
    p_test = os.path.join(processed_dir, "ml_corpus_test_split.csv")

    df_train_split.to_csv(p_train, index=False)
    df_val_split.to_csv(p_val, index=False)
    df_test_split.to_csv(p_test, index=False)

    print(f"Saved: {p_train} ({len(df_train_split)} rows, {(len(df_train_split)/len(df_clean_pool))*100:.2f}%)")
    print(f"Saved: {p_val} ({len(df_val_split)} rows, {(len(df_val_split)/len(df_clean_pool))*100:.2f}%)")
    print(f"Saved: {p_test} ({len(df_test_split)} rows, {(len(df_test_split)/len(df_clean_pool))*100:.2f}%)")
    print("=" * 70)
    print("ML DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    prepare_ml_splits()
