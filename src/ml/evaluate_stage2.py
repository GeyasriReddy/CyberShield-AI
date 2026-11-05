import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def evaluate_stage2(
    model_path="models/stage2_best_model.joblib",
    test_split_path="data/processed/stage2_test_split.csv",
    train_split_path="data/processed/stage2_train_split.csv",
    val_split_path="data/processed/stage2_val_split.csv",
    output_dir="results",
    model_dir="models"
):
    """
    Stage 2 Model Evaluation Pipeline on Untouched Test Set.

    Methodological Notes:
    - Probabilities: Outputs are raw PREDICTED DECISION PROBABILITIES (from tree-voting in
      Random Forest or sigmoid in Logistic Regression). No secondary calibration procedure
      (e.g., Platt scaling or isotonic calibration) was applied at Stage 2.
    - Evaluation Context: This evaluation operates in a TRANSDUCTIVE network setting where SNA
      metrics were computed from the full observed communication graph. The split is strictly
      PAIR-DISJOINT (no shared pair history between splits), but NOT user-disjoint.
    """
    print("=" * 75)
    print("CYBERSHIELD AI — PHASE 6: STAGE 2 MODEL EVALUATION & INFERENCE")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load artifacts
    print("\n>>> Step 1: Loading best model artifact and untouched test set...")
    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle["feature_names"]
    model_name = model_bundle["model_name"]
    features_config = model_bundle.get("features_config", "unknown")
    architecture = model_bundle.get("architecture", "unknown")

    df_test = pd.read_csv(test_split_path)
    print(f"  Loaded model: {model_name} (Architecture: {architecture}, Config: {features_config})")
    print(f"  Model Feature Count ({len(feature_names)}): {feature_names}")
    print(f"  Test set records: {len(df_test):,} directed dyads")

    X_test = df_test[feature_names]
    y_test = df_test["CB_Label"].values

    # 2. Predict on Test Set
    print("\n>>> Step 2: Generating predicted cyberbullying probabilities (uncalibrated decision probabilities)...")
    p_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_50 = (p_test >= 0.50).astype(int)

    # 3. Calculate Comprehensive Evaluation Metrics
    acc = accuracy_score(y_test, y_pred_50)
    prec = precision_score(y_test, y_pred_50, zero_division=0)
    rec = recall_score(y_test, y_pred_50)
    f1 = f1_score(y_test, y_pred_50)
    macro_f1 = f1_score(y_test, y_pred_50, average="macro")
    roc_auc = roc_auc_score(y_test, p_test)
    pr_auc = average_precision_score(y_test, p_test)
    cm = confusion_matrix(y_test, y_pred_50)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 50)
    print("FINAL TEST SET PERFORMANCE (Untouched Evaluation):")
    print("-" * 50)
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision:     {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:        {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:      {f1:.4f}")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"  ROC-AUC:       {roc_auc:.4f}")
    print(f"  PR-AUC:        {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {tn:,} (Correctly classified non-cyberbullying)")
    print(f"  False Positives (FP): {fp:,} (Non-cyberbullying flagged as cyberbullying)")
    print(f"  False Negatives (FN): {fn:,} (Cyberbullying missed)")
    print(f"  True Positives (TP):  {tp:,} (Correctly identified cyberbullying)")

    print("\nClass-Wise Classification Report:")
    report = classification_report(y_test, y_pred_50, target_names=["Non-Cyberbullying (0)", "Cyberbullying (1)"], digits=4)
    print(report)

    # 4. Save Predictions
    pred_df = df_test[["user1_id", "user2_id"]].copy()
    pred_df["true_cb_label"] = y_test
    pred_df["predicted_cb_label"] = y_pred_50
    pred_df["cyberbullying_probability"] = np.round(p_test, 6)

    pred_out_path = os.path.join(output_dir, "stage2_predictions.csv")
    pred_df.to_csv(pred_out_path, index=False)
    print(f">>> Step 3: Saved test predictions: {pred_out_path}")

    # 5. Plot and Save Confusion Matrix
    print("\n>>> Step 4: Generating confusion matrix plot...")
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Stage 2 Cyberbullying Confusion Matrix\n{model_name} (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    classes = ["Non-Bullying (0)", "Cyberbullying (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / len(y_test) * 100
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}\n({pct:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=12, fontweight="bold", color=color)

    plt.ylabel("True Dyad Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Dyad Label", fontsize=12, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "stage2_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix plot: {cm_path}")

    # 6. Evaluate all 4 models on Test Set for Complete Cross-Model Table
    print("\n>>> Step 5: Computing test performance across all 4 architectures (M1, M2, M3, M4)...")
    df_val = pd.read_csv(val_split_path)

    m1_feats = [
        "total_messages", "reciprocal_messages", "reciprocity_ratio",
        "directional_communication_imbalance", "sender_out_strength", "receiver_in_strength"
    ]
    m2_feats = [
        "mean_aggression_probability", "predicted_aggression_volume", "max_aggression_probability"
    ]
    m3_feats = [
        "total_messages", "reciprocal_messages", "reciprocity_ratio",
        "directional_communication_imbalance", "interaction_span_days",
        "inter_message_time_mean_hours", "inter_message_time_std_hours",
        "sender_out_strength", "receiver_in_strength",
        "sender_pagerank", "receiver_pagerank",
        "sender_hub_score", "receiver_authority_score"
    ]
    m4_feats = [
        "total_messages", "reciprocal_messages", "reciprocity_ratio",
        "directional_communication_imbalance", "interaction_span_days",
        "inter_message_time_mean_hours", "inter_message_time_std_hours",
        "mean_aggression_probability", "predicted_aggression_volume", "max_aggression_probability",
        "sender_outgoing_aggression_volume", "receiver_incoming_aggression_exposure",
        "sender_out_strength", "receiver_in_strength",
        "sender_pagerank", "receiver_pagerank",
        "sender_hub_score", "receiver_authority_score"
    ]

    test_configs = {
        "M1_Baseline": m1_feats,
        "M2_NLP_Only": m2_feats,
        "M3_SNA_Only": m3_feats,
        "M4_Combined": m4_feats
    }

    full_comparison = []

    for cfg_name, feats in test_configs.items():
        for arch in ["Logistic Regression", "Random Forest"]:
            slug = f"{cfg_name}_{arch}".lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            artifact_file = os.path.join(model_dir, f"stage2_{slug}.joblib")

            if os.path.exists(artifact_file):
                # Load saved bundle and verify its features
                bundle = joblib.load(artifact_file)
                pipe = bundle["pipeline"]
                loaded_feats = bundle["feature_names"]
                assert loaded_feats == feats, f"Feature mismatch in {artifact_file}"
            else:
                # Fallback to refit if artifact not yet saved
                df_train = pd.read_csv(train_split_path)
                if arch == "Logistic Regression":
                    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
                else:
                    clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                pipe.fit(df_train[feats], df_train["CB_Label"])

            # Val metrics
            p_val = pipe.predict_proba(df_val[feats])[:, 1]
            pred_v = (p_val >= 0.5).astype(int)

            # Test metrics
            p_t = pipe.predict_proba(df_test[feats])[:, 1]
            pred_t = (p_t >= 0.5).astype(int)

            full_comparison.append({
                "model_name": f"{cfg_name} ({arch})",
                "feature_set": cfg_name,
                "architecture": arch,
                "num_features": len(feats),
                "val_accuracy": round(accuracy_score(df_val["CB_Label"], pred_v), 4),
                "val_f1": round(f1_score(df_val["CB_Label"], pred_v), 4),
                "val_pr_auc": round(average_precision_score(df_val["CB_Label"], p_val), 4),
                "val_roc_auc": round(roc_auc_score(df_val["CB_Label"], p_val), 4),
                "test_accuracy": round(accuracy_score(y_test, pred_t), 4),
                "test_precision": round(precision_score(y_test, pred_t, zero_division=0), 4),
                "test_recall": round(recall_score(y_test, pred_t), 4),
                "test_f1": round(f1_score(y_test, pred_t), 4),
                "test_macro_f1": round(f1_score(y_test, pred_t, average="macro"), 4),
                "test_pr_auc": round(average_precision_score(y_test, p_t), 4),
                "test_roc_auc": round(roc_auc_score(y_test, p_t), 4)
            })

    full_comp_df = pd.DataFrame(full_comparison)
    full_comp_path = os.path.join(output_dir, "stage2_model_comparison.csv")
    full_comp_df.to_csv(full_comp_path, index=False)
    print(f"  Updated complete comparison: {full_comp_path}")
    print("\nComplete Cross-Model Evaluation (Val vs Test):")
    print(full_comp_df[["model_name", "num_features", "val_f1", "val_pr_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_pr_auc", "test_roc_auc"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("STAGE 2 EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    return full_comp_df

if __name__ == "__main__":
    evaluate_stage2()
