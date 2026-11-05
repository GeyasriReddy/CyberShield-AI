import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

def calibrate_model(estimator, X_train, y_train, cv=3, method="sigmoid", sample_weight=None):
    """
    Calibrates probability estimates using cross-validation on training data only.
    Prevents raw SVM decision scores from being erroneously treated as probabilities.
    
    Parameters:
    -----------
    estimator : estimator instance
        The base classifier (e.g. LinearSVC or LogisticRegression)
    X_train : sparse matrix or array-like
        Training features
    y_train : array-like
        Training labels
    cv : int
        Number of cross-validation folds (default 3)
    method : str
        'sigmoid' (Platt scaling) or 'isotonic'
    sample_weight : array-like, optional
        Sample weights for training
    """
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv
    )
    if sample_weight is not None:
        calibrated.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        calibrated.fit(X_train, y_train)
    return calibrated

def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Computes Brier score and calibration curve data points.
    
    Parameters:
    -----------
    y_true : array-like
        Binary ground truth labels (0 or 1)
    y_prob : array-like
        Predicted probabilities for positive class (1)
    n_bins : int
        Number of probability bins
        
    Returns:
    --------
    dict containing:
        - 'brier_score': float
        - 'prob_true': array
        - 'prob_pred': array
    """
    brier = brier_score_loss(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "brier_score": float(brier),
        "prob_true": prob_true,
        "prob_pred": prob_pred
    }

def plot_calibration_curves(models_dict, X_test, y_test, output_path="results/calibration_curves.png"):
    """
    Plots reliability curves for multiple models against the perfectly calibrated diagonal.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            cal_data = evaluate_calibration(y_test, probs)
            plt.plot(
                cal_data["prob_pred"],
                cal_data["prob_true"],
                marker="o",
                label=f"{name} (Brier={cal_data['brier_score']:.4f})"
            )

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of True Positives", fontsize=11)
    plt.title("Reliability Calibration Curves (Aggression Detection)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved calibration curves: {output_path}")

