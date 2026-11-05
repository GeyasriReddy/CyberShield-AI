import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def extract_model_feature_importance(pipeline, feature_names):
    """
    Extracts model-specific feature importance matching the model's exact features.
    - Random Forest: feature_importances_ (Gini impurity decrease)
    - Logistic Regression: standardized coefficients (coef_[0]) and abs(coef_[0])
    """
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        df_fi = pd.DataFrame({
            "feature_name": list(feature_names),
            "importance_score": np.round(importances, 6),
            "importance_metric": "gini_importance"
        }).sort_values(by="importance_score", ascending=False).reset_index(drop=True)
        df_fi["rank"] = range(1, len(df_fi) + 1)
        return df_fi
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        df_fi = pd.DataFrame({
            "feature_name": list(feature_names),
            "standardized_coef": np.round(coefs, 6),
            "abs_coef": np.round(np.abs(coefs), 6),
            "importance_score": np.round(np.abs(coefs), 6),
            "importance_metric": "abs_standardized_coefficient"
        }).sort_values(by="importance_score", ascending=False).reset_index(drop=True)
        df_fi["rank"] = range(1, len(df_fi) + 1)
        return df_fi
    else:
        raise ValueError(f"Classifier {type(clf)} does not support feature importance extraction")

def train_stage2_models(
    dyads_path="data/processed/dyad_sna_features.csv",
    cb_labels_path="data/processed/cb_labels_cleaned.csv",
    output_dir="results",
    model_dir="models",
    processed_dir="data/processed",
    random_state=42
):
    """
    Stage 2 Cyberbullying Detection Training Pipeline.

    Methodological Note on Evaluation Setting:
    This experiment operates in a TRANSDUCTIVE network setting. Social Network Analysis (SNA)
    metrics were extracted from the full observed communication graph across the 100 students.
    The evaluation split is strictly PAIR-DISJOINT: both directed edges between user u and user v
    (u -> v and v -> u) are placed in the same split, ensuring zero reverse-dyad leakage.
    However, it is NOT user-disjoint, as all 100 students have other dyadic interactions across
    splits. This model predicts dyadic relationship classification within the monitored cohort;
    it does not claim zero-shot generalization to an entirely unobserved school.
    """
    print("=" * 75)
    print("CYBERSHIELD AI — PHASE 6: STAGE 2 CYBERBULLYING MODEL TRAINING")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load Datasets
    print("\n>>> Step 1: Loading verified datasets and validating join keys...")
    df_dyads = pd.read_csv(dyads_path)
    df_cb = pd.read_csv(cb_labels_path)

    # Join strictly on verified User ID pairs
    merged = pd.merge(
        df_dyads,
        df_cb[["User1 ID", "User2 ID", "CB_Label"]],
        left_on=["user1_id", "user2_id"],
        right_on=["User1 ID", "User2 ID"],
        how="inner"
    )

    if len(merged) != len(df_dyads) or len(merged) != len(df_cb):
        raise ValueError(f"Dyad join mismatch: merged={len(merged)}, dyads={len(df_dyads)}, cb={len(df_cb)}")

    print(f"  Successfully joined {len(merged):,} directed dyads with ground truth labels.")
    print(f"  Class 0 (Non-Cyberbullying): {(merged['CB_Label'] == 0).sum():,} ({((merged['CB_Label'] == 0).mean()*100):.2f}%)")
    print(f"  Class 1 (Cyberbullying):     {(merged['CB_Label'] == 1).sum():,} ({((merged['CB_Label'] == 1).mean()*100):.2f}%)")

    # 2. Pair-Grouped Stratified Data Splitting (Leakage-Free)
    print("\n>>> Step 2: Constructing pair-grouped stratified splits (70/15/15)...")
    # Grouping by unordered pair guarantees zero reverse-dyad leakage across splits
    merged["pair_id"] = merged.apply(
        lambda r: f"{min(int(r['user1_id']), int(r['user2_id']))}_{max(int(r['user1_id']), int(r['user2_id']))}",
        axis=1
    )

    pair_labels = merged.groupby("pair_id")["CB_Label"].max()
    pair_df = pd.DataFrame({"pair_id": pair_labels.index, "pair_label": pair_labels.values})

    train_pairs, temp_pairs = train_test_split(
        pair_df, test_size=0.30, random_state=random_state, stratify=pair_df["pair_label"]
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.50, random_state=random_state, stratify=temp_pairs["pair_label"]
    )

    train_set = set(train_pairs["pair_id"])
    val_set = set(val_pairs["pair_id"])
    test_set = set(test_pairs["pair_id"])

    # Verify zero pair overlap
    assert len(train_set.intersection(val_set)) == 0, "Train-Val pair leakage detected!"
    assert len(train_set.intersection(test_set)) == 0, "Train-Test pair leakage detected!"
    assert len(val_set.intersection(test_set)) == 0, "Val-Test pair leakage detected!"

    df_train = merged[merged["pair_id"].isin(train_set)].copy()
    df_val = merged[merged["pair_id"].isin(val_set)].copy()
    df_test = merged[merged["pair_id"].isin(test_set)].copy()

    # Save split datasets
    train_split_path = os.path.join(processed_dir, "stage2_train_split.csv")
    val_split_path = os.path.join(processed_dir, "stage2_val_split.csv")
    test_split_path = os.path.join(processed_dir, "stage2_test_split.csv")

    df_train.to_csv(train_split_path, index=False)
    df_val.to_csv(val_split_path, index=False)
    df_test.to_csv(test_split_path, index=False)

    print(f"  Train Split: {len(df_train):,} dyads (Positive: {(df_train['CB_Label'] == 1).sum()} | {((df_train['CB_Label'] == 1).mean()*100):.2f}%) -> {train_split_path}")
    print(f"  Val Split:   {len(df_val):,} dyads (Positive: {(df_val['CB_Label'] == 1).sum()} | {((df_val['CB_Label'] == 1).mean()*100):.2f}%) -> {val_split_path}")
    print(f"  Test Split:  {len(df_test):,} dyads (Positive: {(df_test['CB_Label'] == 1).sum()} | {((df_test['CB_Label'] == 1).mean()*100):.2f}%) -> {test_split_path}")

    # 3. Define Feature Subsets for M1, M2, M3, M4
    print("\n>>> Step 3: Defining controlled feature subsets (M1, M2, M3, M4)...")
    m1_feats = [
        "total_messages",
        "reciprocal_messages",
        "reciprocity_ratio",
        "directional_communication_imbalance",
        "sender_out_strength",
        "receiver_in_strength"
    ]

    m2_feats = [
        "mean_aggression_probability",
        "predicted_aggression_volume",
        "max_aggression_probability"
    ]

    m3_feats = [
        "total_messages",
        "reciprocal_messages",
        "reciprocity_ratio",
        "directional_communication_imbalance",
        "interaction_span_days",
        "inter_message_time_mean_hours",
        "inter_message_time_std_hours",
        "sender_out_strength",
        "receiver_in_strength",
        "sender_pagerank",
        "receiver_pagerank",
        "sender_hub_score",
        "receiver_authority_score"
    ]

    m4_feats = [
        "total_messages",
        "reciprocal_messages",
        "reciprocity_ratio",
        "directional_communication_imbalance",
        "interaction_span_days",
        "inter_message_time_mean_hours",
        "inter_message_time_std_hours",
        "mean_aggression_probability",
        "predicted_aggression_volume",
        "max_aggression_probability",
        "sender_outgoing_aggression_volume",
        "receiver_incoming_aggression_exposure",
        "sender_out_strength",
        "receiver_in_strength",
        "sender_pagerank",
        "receiver_pagerank",
        "sender_hub_score",
        "receiver_authority_score"
    ]

    # Target leakage verification across ALL individual feature subsets (M1, M2, M3, M4)
    forbidden_terms = ["cb_label", "aggressive_count", "intent_to_harm", "peerness", "exploratory_peerness_score"]
    feature_configs_to_verify = [
        ("M1_Baseline", m1_feats),
        ("M2_NLP_Only", m2_feats),
        ("M3_SNA_Only", m3_feats),
        ("M4_Combined", m4_feats)
    ]
    for config_name, feats in feature_configs_to_verify:
        for f in feats:
            for forbidden in forbidden_terms:
                if forbidden in f.lower():
                    raise ValueError(f"CRITICAL LEAKAGE: {config_name} feature '{f}' contains forbidden substring '{forbidden}'")

    print(f"  M1 (Interaction Baseline): {len(m1_feats)} features (Verified leakage-free)")
    print(f"  M2 (NLP Aggression Only):  {len(m2_feats)} features (Verified leakage-free)")
    print(f"  M3 (SNA & Temporal Only):  {len(m3_feats)} features (Verified leakage-free)")
    print(f"  M4 (Combined NLP + SNA):   {len(m4_feats)} features (Verified leakage-free)")
    print("  Zero target leakage columns detected across all 4 individual feature sets.")

    y_train = df_train["CB_Label"].values
    y_val = df_val["CB_Label"].values

    models_config = {
        "M1_Baseline": m1_feats,
        "M2_NLP_Only": m2_feats,
        "M3_SNA_Only": m3_feats,
        "M4_Combined": m4_feats
    }

    comparison_records = []
    fitted_models = {}
    saved_artifacts = {}

    print("\n>>> Step 4: Training, evaluating, and exporting all 8 model configurations...")
    t_start = time.time()

    for config_name, feats in models_config.items():
        X_tr = df_train[feats]
        X_va = df_val[feats]

        # A. Logistic Regression Pipeline
        lr_name = f"{config_name} (Logistic Regression)"
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state))
        ])
        lr_pipe.fit(X_tr, y_train)
        p_val_lr = lr_pipe.predict_proba(X_va)[:, 1]
        pred_lr = (p_val_lr >= 0.5).astype(int)

        fitted_models[lr_name] = lr_pipe
        lr_record = {
            "model_name": lr_name,
            "architecture": "Logistic Regression",
            "feature_set": config_name,
            "num_features": len(feats),
            "val_accuracy": round(accuracy_score(y_val, pred_lr), 4),
            "val_precision": round(precision_score(y_val, pred_lr, zero_division=0), 4),
            "val_recall": round(recall_score(y_val, pred_lr), 4),
            "val_f1": round(f1_score(y_val, pred_lr), 4),
            "val_macro_f1": round(f1_score(y_val, pred_lr, average="macro"), 4),
            "val_roc_auc": round(roc_auc_score(y_val, p_val_lr), 4),
            "val_pr_auc": round(average_precision_score(y_val, p_val_lr), 4)
        }
        comparison_records.append(lr_record)

        # Save individual model bundle with exact feature list
        lr_slug = lr_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        lr_path = os.path.join(model_dir, f"stage2_{lr_slug}.joblib")
        joblib.dump({
            "pipeline": lr_pipe,
            "model_name": lr_name,
            "architecture": "Logistic Regression",
            "features_config": config_name,
            "feature_names": list(feats),
            "num_features": len(feats),
            "random_state": random_state,
            "probability_type": "predicted_probability_uncalibrated",
            "val_metrics": lr_record
        }, lr_path)
        saved_artifacts[lr_name] = lr_path

        # B. Random Forest Pipeline
        rf_name = f"{config_name} (Random Forest)"
        rf_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=3,
                class_weight="balanced", random_state=random_state, n_jobs=-1
            ))
        ])
        rf_pipe.fit(X_tr, y_train)
        p_val_rf = rf_pipe.predict_proba(X_va)[:, 1]
        pred_rf = (p_val_rf >= 0.5).astype(int)

        fitted_models[rf_name] = rf_pipe
        rf_record = {
            "model_name": rf_name,
            "architecture": "Random Forest",
            "feature_set": config_name,
            "num_features": len(feats),
            "val_accuracy": round(accuracy_score(y_val, pred_rf), 4),
            "val_precision": round(precision_score(y_val, pred_rf, zero_division=0), 4),
            "val_recall": round(recall_score(y_val, pred_rf), 4),
            "val_f1": round(f1_score(y_val, pred_rf), 4),
            "val_macro_f1": round(f1_score(y_val, pred_rf, average="macro"), 4),
            "val_roc_auc": round(roc_auc_score(y_val, p_val_rf), 4),
            "val_pr_auc": round(average_precision_score(y_val, p_val_rf), 4)
        }
        comparison_records.append(rf_record)

        # Save individual model bundle with exact feature list
        rf_slug = rf_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        rf_path = os.path.join(model_dir, f"stage2_{rf_slug}.joblib")
        joblib.dump({
            "pipeline": rf_pipe,
            "model_name": rf_name,
            "architecture": "Random Forest",
            "features_config": config_name,
            "feature_names": list(feats),
            "num_features": len(feats),
            "random_state": random_state,
            "probability_type": "predicted_probability_uncalibrated",
            "val_metrics": rf_record
        }, rf_path)
        saved_artifacts[rf_name] = rf_path

    df_comp = pd.DataFrame(comparison_records)
    comp_path = os.path.join(output_dir, "stage2_model_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"  Model comparison saved: {comp_path}")
    print("\nValidation Performance Summary:")
    print(df_comp[["model_name", "num_features", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_pr_auc", "val_roc_auc"]].to_string(index=False))

    # 5. Best Model Selection & Export
    # Select best model strictly using Validation F1 (Test set is strictly descriptive)
    best_row = df_comp.loc[df_comp["val_f1"].idxmax()]
    best_name = best_row["model_name"]
    best_config_name = best_row["feature_set"]
    best_arch = best_row["architecture"]
    best_feats = list(models_config[best_config_name])
    best_pipeline = fitted_models[best_name]

    best_model_path = os.path.join(model_dir, "stage2_best_model.joblib")
    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "architecture": best_arch,
        "features_config": best_config_name,
        "feature_names": best_feats,
        "num_features": len(best_feats),
        "random_state": random_state,
        "probability_type": "predicted_probability_uncalibrated",
        "val_metrics": best_row.to_dict()
    }, best_model_path)
    print(f"\n>>> Step 5: Exported Best Model Artifact: '{best_name}' -> {best_model_path}")
    print(f"  Correct Feature List ({len(best_feats)} features): {best_feats}")

    # 6. Feature Importance Extraction (Model-Specific & Exact Feature List)
    print("\n>>> Step 6: Extracting model-specific feature importance for best model...")
    fi_df = extract_model_feature_importance(best_pipeline, best_feats)
    fi_path = os.path.join(output_dir, "stage2_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"  Feature importance saved: {fi_path}")
    print(f"\nTop Features by Model-Specific Metric ({best_arch}):")
    print(fi_df.to_string(index=False))

    print(f"\nStage 2 Training completed in {time.time()-t_start:.2f}s.")
    print("=" * 75)
    return best_pipeline, df_comp, fi_df

if __name__ == "__main__":
    train_stage2_models()
