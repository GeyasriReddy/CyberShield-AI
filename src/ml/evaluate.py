import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve
)

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes all standard performance metrics for binary classification.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    if y_prob is not None:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["pr_auc"] = np.nan
        metrics["roc_auc"] = np.nan
        metrics["brier_score"] = np.nan

    return metrics

def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """
    Plots a clear, professional confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / np.sum(cm)) * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Aggressive (0)", "Aggressive (1)"], fontsize=10)
    ax.set_yticklabels(["Non-Aggressive (0)", "Aggressive (1)"], fontsize=10)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.ylabel("True Class", fontsize=11)
    plt.title(f"Confusion Matrix: {title}", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")

def plot_pr_curves(models_dict, X_test, y_test, output_path="results/precision_recall_curve.png"):
    """
    Plots Precision-Recall curves for all probabilistic/calibrated models.
    """
    plt.figure(figsize=(8, 6))

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    # Baseline no-skill PR line (prevalence of positive class)
    no_skill = np.mean(y_test)
    plt.axhline(y=no_skill, color="gray", linestyle="--", label=f"No-Skill Baseline ({no_skill:.2f})")

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves (Aggressive Language)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall curves: {output_path}")

