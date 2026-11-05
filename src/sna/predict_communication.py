import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_data_prep import cybershield_url_normalizer

def generate_communication_predictions(
    comm_path="data/processed/communication_cleaned_sorted.csv",
    model_path="models/best_model_pipeline.joblib",
    output_path="data/processed/communication_with_predictions.csv",
    batch_size=10000
):
    print("=" * 70)
    print("CYBERSHIELD AI — STEP 2: PREDICTING COMMUNICATION AGGRESSION")
    print("=" * 70)

    # 1. Load data & model
    print(f"Loading communication data from: {comm_path}")
    df_comm = pd.read_csv(comm_path)
    total_rows = len(df_comm)
    print(f"Total communication events: {total_rows:,}")

    print(f"Loading trained model pipeline from: {model_path}")
    pipeline = joblib.load(model_path)

    # Check for missing messages
    null_messages = df_comm["Message"].isna().sum()
    if null_messages > 0:
        raise ValueError(f"Found {null_messages} missing/NaN messages in communication dataset!")

    # Check empty string messages
    empty_messages = (df_comm["Message"].astype(str).str.strip() == "").sum()
    if empty_messages > 0:
        raise ValueError(f"Found {empty_messages} empty string messages in communication dataset!")

    # 2. Batch Inference
    print(f"\nProcessing {total_rows:,} messages in batches of {batch_size:,}...")
    t0 = time.time()
    all_probs = []
    all_preds = []

    for i in range(0, total_rows, batch_size):
        batch_texts = df_comm["Message"].iloc[i : i + batch_size].tolist()
        batch_probs = pipeline.predict_proba(batch_texts)[:, 1] # Probability of Class 1 (Aggressive)
        batch_preds = pipeline.predict(batch_texts)

        all_probs.extend(batch_probs)
        all_preds.extend(batch_preds)
        processed = min(i + batch_size, total_rows)
        print(f"  Processed {processed:,} / {total_rows:,} messages ({(processed/total_rows)*100:.1f}%) in {time.time()-t0:.1f}s")

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # 3. Validation of Predictions
    print("\nValidating predictions...")
    assert len(all_probs) == total_rows, "Prediction count mismatch!"
    assert (all_probs >= 0.0).all() and (all_probs <= 1.0).all(), "Invalid probabilities outside [0, 1]!"
    print(f"Probability range: min={all_probs.min():.6f}, mean={all_probs.mean():.4f}, max={all_probs.max():.6f}")

    # 4. Determine prediction status
    # Differentiates standard unambiguous messages vs ambiguous overlap messages
    pred_status = np.where(
        df_comm["is_ambiguous_overlap"],
        "ambiguous_overlap_prediction",
        "standard_prediction"
    )

    # 5. Assemble and save output
    df_out = df_comm.copy()
    df_out["aggression_probability"] = np.round(all_probs, 6)
    df_out["aggression_prediction"] = all_preds.astype(int)
    df_out["prediction_status"] = pred_status

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved annotated communication data: {output_path} ({len(df_out):,} rows)")
    print("=" * 70)

    # Summary distribution
    print("\nPrediction Summary:")
    print(df_out["aggression_prediction"].value_counts())
    print("\nPrediction Status breakdown:")
    print(df_out["prediction_status"].value_counts())
    print("=" * 70)

    return df_out

if __name__ == "__main__":
    generate_communication_predictions()
