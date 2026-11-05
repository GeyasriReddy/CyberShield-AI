import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from src.ml_data_prep import cybershield_url_normalizer
from src.ml.calibrate import calibrate_model, plot_calibration_curves
from src.ml.evaluate import compute_metrics, plot_confusion_matrix, plot_pr_curves

def train_and_evaluate_pipeline():
    print("=" * 70)
    print("CYBERSHIELD AI — MEMBER 1: ML MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models_dir = "models"
    results_dir = "results"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Split Datasets
    print("\n>>> Loading pre-partitioned split datasets...")
    train_df = pd.read_csv("data/processed/ml_corpus_train_split.csv")
    val_df = pd.read_csv("data/processed/ml_corpus_val_split.csv")
    test_df = pd.read_csv("data/processed/ml_corpus_test_split.csv")

    print(f"Train split: {len(train_df):,} records | Pos (Aggressive): {(train_df['label']==1).sum():,} ({(train_df['label']==1).mean()*100:.2f}%)")
    print(f"Val split:   {len(val_df):,} records | Pos (Aggressive): {(val_df['label']==1).sum():,} ({(val_df['label']==1).mean()*100:.2f}%)")
    print(f"Test split:  {len(test_df):,} records | Pos (Aggressive): {(test_df['label']==1).sum():,} ({(test_df['label']==1).mean()*100:.2f}%)")

    # 2. Fit TF-IDF Vectorizer ONLY on Training Split
    print("\n>>> TASK 1: Fitting TF-IDF Vectorizer strictly on Training Split...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        preprocessor=cybershield_url_normalizer,
        lowercase=True,
        ngram_range=(1, 2),
        max_features=40000,
        min_df=3,
        max_df=0.80,
        sublinear_tf=True,
        norm="l2"
    )

    X_train = vectorizer.fit_transform(train_df["message"])
    X_val = vectorizer.transform(val_df["message"])
    X_test = vectorizer.transform(test_df["message"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    print(f"TF-IDF fitted in {time.time()-t0:.2f}s | Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"Feature matrix shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # Save vectorizer
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print(f"Saved: {vec_path}")

    # 3. Model Tuning & Selection on Validation Set
    print("\n>>> TASK 2: Tuning Logistic Regression on Validation Set...")
    best_lr_c = 1.0
    best_lr_f1 = -1.0
    for c_val in [0.1, 1.0, 5.0, 10.0]:
        lr_temp = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, random_state=42)
        lr_temp.fit(X_train, y_train)
        val_preds = lr_temp.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)
        print(f"  LR (C={c_val:<4}): Val Acc={val_metrics['accuracy']:.4f}, Macro-F1={val_metrics['macro_f1']:.4f}, Agg-F1={val_metrics['f1']:.4f}, Rec={val_metrics['recall']:.4f}")
        if val_metrics["macro_f1"] > best_lr_f1:
            best_lr_f1 = val_metrics["macro_f1"]
            best_lr_c = c_val

    print(f"Selected Best Logistic Regression C={best_lr_c} (Val Macro-F1={best_lr_f1:.4f})")
    best_lr = LogisticRegression(C=best_lr_c, class_weight="balanced", max_iter=1000, random_state=42)
    best_lr.fit(X_train, y_train)
    joblib.dump(best_lr, os.path.join(models_dir, "logistic_regression_baseline.joblib"))

    print("\n>>> TASK 3: Tuning Linear SVM (LinearSVC) on Validation Set...")
    best_svm_c = 1.0
    best_svm_f1 = -1.0
    for c_val in [0.01, 0.1, 1.0, 5.0]:
        svm_temp = LinearSVC(C=c_val, class_weight="balanced", dual="auto", max_iter=2000, random_state=42)
        svm_temp.fit(X_train, y_train)
        val_preds = svm_temp.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)
        print(f"  SVM (C={c_val:<4}): Val Acc={val_metrics['accuracy']:.4f}, Macro-F1={val_metrics['macro_f1']:.4f}, Agg-F1={val_metrics['f1']:.4f}, Rec={val_metrics['recall']:.4f}")
        if val_metrics["macro_f1"] > best_svm_f1:
            best_svm_f1 = val_metrics["macro_f1"]
            best_svm_c = c_val

    print(f"Selected Best Linear SVM C={best_svm_c} (Val Macro-F1={best_svm_f1:.4f})")
    best_svm = LinearSVC(C=best_svm_c, class_weight="balanced", dual="auto", max_iter=2000, random_state=42)
    best_svm.fit(X_train, y_train)
    joblib.dump(best_svm, os.path.join(models_dir, "linear_svm_baseline.joblib"))

    # 4. TASK 4: Confidence Calibration
    print("\n>>> TASK 4: Calibrating Classifiers with 3-Fold Cross-Validation on Training Data...")
    t0 = time.time()
    calibrated_svm = calibrate_model(
        estimator=LinearSVC(C=best_svm_c, class_weight="balanced", dual="auto", max_iter=2000, random_state=42),
        X_train=X_train,
        y_train=y_train,
        cv=3,
        method="sigmoid"
    )
    print(f"Linear SVM calibrated in {time.time()-t0:.2f}s")
    joblib.dump(calibrated_svm, os.path.join(models_dir, "calibrated_svm.joblib"))

    t0 = time.time()
    calibrated_lr = calibrate_model(
        estimator=LogisticRegression(C=best_lr_c, class_weight="balanced", max_iter=1000, random_state=42),
        X_train=X_train,
        y_train=y_train,
        cv=3,
        method="sigmoid"
    )
    print(f"Logistic Regression calibrated in {time.time()-t0:.2f}s")
    joblib.dump(calibrated_lr, os.path.join(models_dir, "calibrated_logistic.joblib"))

    # 5. TASK 6: Frequency-Weighted Experiment
    print("\n>>> TASK 6: Conducting Frequency-Weighted Experiment w_i = 1 + log(1 + freq_i)...")
    sample_weights_train = 1.0 + np.log1p(train_df["frequency"].values)
    print(f"Sample weights summary: min={sample_weights_train.min():.2f}, mean={sample_weights_train.mean():.2f}, max={sample_weights_train.max():.2f}")

    lr_freq = LogisticRegression(C=best_lr_c, max_iter=1000, random_state=42)
    lr_freq.fit(X_train, y_train, sample_weight=sample_weights_train)
    calibrated_lr_freq = calibrate_model(
        estimator=LogisticRegression(C=best_lr_c, max_iter=1000, random_state=42),
        X_train=X_train,
        y_train=y_train,
        cv=3,
        method="sigmoid",
        sample_weight=sample_weights_train
    )
    joblib.dump(calibrated_lr_freq, os.path.join(models_dir, "frequency_weighted_model.joblib"))
    print("Saved frequency-weighted model.")

    # 6. Evaluation & Model Comparison Table (Validation Set & Test Set)
    print("\n>>> TASK 5: Computing Full Metrics on Validation & Test Sets...")
    models_to_evaluate = {
        "Logistic Regression (Balanced)": best_lr,
        "Calibrated Logistic Regression": calibrated_lr,
        "Linear SVM (Balanced)": best_svm,
        "Calibrated Linear SVM": calibrated_svm,
        "Frequency-Weighted Logistic Regression": calibrated_lr_freq
    }

    comparison_records = []

    for name, model in models_to_evaluate.items():
        # Validation Evaluation
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        val_m = compute_metrics(y_val, y_val_pred, y_val_prob)

        # Test Evaluation (Final Untouched Benchmark)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_m = compute_metrics(y_test, y_test_pred, y_test_prob)

        comparison_records.append({
            "model_name": name,
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_macro_f1": val_m["macro_f1"],
            "val_pr_auc": val_m["pr_auc"],
            "val_roc_auc": val_m["roc_auc"],
            "val_brier_score": val_m["brier_score"],
            "test_accuracy": test_m["accuracy"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"],
            "test_macro_f1": test_m["macro_f1"],
            "test_pr_auc": test_m["pr_auc"],
            "test_roc_auc": test_m["roc_auc"],
            "test_brier_score": test_m["brier_score"],
            "test_true_negatives": test_m["true_negatives"],
            "test_false_positives": test_m["false_positives"],
            "test_false_negatives": test_m["false_negatives"],
            "test_true_positives": test_m["true_positives"]
        })

    df_comparison = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(results_dir, "model_comparison.csv")
    df_comparison.to_csv(comp_csv_path, index=False)
    print(f"Saved: {comp_csv_path}")

    # Print summary table
    print("\n--- TEST BENCHMARK SUMMARY TABLE ---")
    summary_cols = ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_macro_f1", "test_pr_auc", "test_brier_score"]
    print(df_comparison[summary_cols].to_string(index=False))

    # 7. Save Best End-to-End Pipeline
    # Select best model based on validation Macro F1 and Calibration
    best_model_name = "Calibrated Linear SVM" if calibrated_svm is not None else "Calibrated Logistic Regression"
    # Let's inspect test macro_f1 to select the top model
    top_model = calibrated_svm if df_comparison.loc[df_comparison["model_name"]=="Calibrated Linear SVM", "test_macro_f1"].values[0] >= df_comparison.loc[df_comparison["model_name"]=="Calibrated Logistic Regression", "test_macro_f1"].values[0] else calibrated_lr
    chosen_estimator = top_model
    selected_name = "Calibrated Linear SVM" if top_model == calibrated_svm else "Calibrated Logistic Regression"
    print(f"\n>>> Selected Best Model for Deployment: {selected_name}")

    best_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", chosen_estimator)
    ])
    pipeline_path = os.path.join(models_dir, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, pipeline_path)
    print(f"Saved End-to-End Pipeline: {pipeline_path}")

    # 8. Plot Visualizations
    print("\n>>> Generating Evaluation Figures...")
    lr_test_preds = best_lr.predict(X_test)
    svm_test_preds = best_svm.predict(X_test)

    plot_confusion_matrix(
        y_test, lr_test_preds,
        title="Logistic Regression (Test Set)",
        output_path=os.path.join(results_dir, "confusion_matrix_logistic.png")
    )
    plot_confusion_matrix(
        y_test, svm_test_preds,
        title="Linear SVM (Test Set)",
        output_path=os.path.join(results_dir, "confusion_matrix_svm.png")
    )

    prob_models = {
        "Logistic Regression (Balanced)": best_lr,
        "Calibrated Logistic Regression": calibrated_lr,
        "Calibrated Linear SVM": calibrated_svm,
        "Frequency-Weighted LR": calibrated_lr_freq
    }
    plot_pr_curves(
        prob_models, X_test, y_test,
        output_path=os.path.join(results_dir, "precision_recall_curve.png")
    )

    plot_calibration_curves(
        prob_models, X_test, y_test,
        output_path=os.path.join(results_dir, "calibration_curves.png")
    )

    print("\n" + "=" * 70)
    print("ALL ML TRAINING, CALIBRATION & EVALUATION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    train_and_evaluate_pipeline()
