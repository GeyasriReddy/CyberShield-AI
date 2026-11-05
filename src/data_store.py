"""
Data Store and Validation Module
--------------------------------
Provides realistic sample datasets, review queue storage, and data validation
for the CyberShield AI moderation platform.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta


def get_initial_sample_comments() -> List[Dict[str, Any]]:
    """
    Returns realistic initial mock comments representing a moderation stream.
    All items are explicitly labeled as sample data.
    """
    now = datetime.now()
    
    sample_records = [
        {
            "id": "CMT-1001",
            "timestamp": (now - timedelta(minutes=12)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_103",
            "comment_text": "Hey check out the updated presentation link for the team meeting tomorrow.",
            "predicted_label": "Clean",
            "confidence": 0.94,
            "recommended_action": "Allow",
            "reason": "Clear constructive intent, no derogatory or aggressive terms.",
            "status": "Auto-Allowed",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1002",
            "timestamp": (now - timedelta(minutes=28)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_107",
            "comment_text": "You are an absolute idiot and everybody on this forum wishes you would leave.",
            "predicted_label": "Toxic",
            "confidence": 0.89,
            "recommended_action": "Flag",
            "reason": "Targeted insult directed at an individual with derogatory terminology.",
            "status": "Pending Review",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1003",
            "timestamp": (now - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_110",
            "comment_text": "Keep posting and see what happens to your account. I will ruin you.",
            "predicted_label": "Harassment",
            "confidence": 0.91,
            "recommended_action": "Flag",
            "reason": "Direct intimidation and harassment pattern detected.",
            "status": "Pending Review",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1004",
            "timestamp": (now - timedelta(hours=1, minutes=10)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_105",
            "comment_text": "Your code is completely pathetic, whoever wrote this should be fired immediately.",
            "predicted_label": "Toxic",
            "confidence": 0.74,
            "recommended_action": "Review",
            "reason": "Harsh criticism bordering on toxicity. Moderator nuance needed.",
            "status": "Pending Review",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1005",
            "timestamp": (now - timedelta(hours=1, minutes=35)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_114",
            "comment_text": "I disagree with this design approach, but thanks for sharing the alternative perspective.",
            "predicted_label": "Clean",
            "confidence": 0.97,
            "recommended_action": "Allow",
            "reason": "Respectful disagreement and constructive dialogue.",
            "status": "Auto-Allowed",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1006",
            "timestamp": (now - timedelta(hours=2, minutes=5)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_102",
            "comment_text": "Nobody asked for your opinion. Just be quiet already.",
            "predicted_label": "Toxic",
            "confidence": 0.68,
            "recommended_action": "Review",
            "reason": "Dismissive and uncivil tone. Human review advised for context.",
            "status": "Pending Review",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1007",
            "timestamp": (now - timedelta(hours=2, minutes=40)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_118",
            "comment_text": "Can someone help me troubleshoot this syntax error in the helper script?",
            "predicted_label": "Clean",
            "confidence": 0.98,
            "recommended_action": "Allow",
            "reason": "Technical inquiry, completely benign.",
            "status": "Auto-Allowed",
            "moderator_notes": ""
        },
        {
            "id": "CMT-1008",
            "timestamp": (now - timedelta(hours=3, minutes=15)).strftime("%Y-%m-%d %H:%M"),
            "user_id": "usr_106",
            "comment_text": "Why do you keep tagging me? Leave me alone or I will report you.",
            "predicted_label": "Harassment",
            "confidence": 0.71,
            "recommended_action": "Review",
            "reason": "Conflict dispute. Might be self-defense rather than harassment source.",
            "status": "Pending Review",
            "moderator_notes": ""
        }
    ]
    return sample_records


def calculate_overview_metrics(comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics for the Overview dashboard.
    """
    total = len(comments)
    recommended_review = sum(1 for c in comments if c.get("recommended_action") == "Review")
    recommended_flag = sum(1 for c in comments if c.get("recommended_action") == "Flag")
    allowed = sum(1 for c in comments if c.get("recommended_action") == "Allow")
    pending_queue = sum(1 for c in comments if c.get("status") == "Pending Review")

    return {
        "total_analyzed": total,
        "recommended_review": recommended_review,
        "recommended_flag": recommended_flag,
        "allowed_count": allowed,
        "pending_queue_count": pending_queue
    }


def validate_comment_input(text: Optional[str]) -> Tuple[bool, str]:
    """
    Validates submitted comment text before running analysis.
    """
    if text is None or not text.strip():
        return False, "Comment text cannot be empty or solely whitespace."
    if len(text.strip()) > 2000:
        return False, "Comment exceeds the maximum allowed length of 2,000 characters."
    return True, ""


def validate_user_id(user_id: Optional[str]) -> Tuple[bool, str]:
    """
    Validates optional user ID string.
    """
    if not user_id or not user_id.strip():
        return True, ""  # Optional field
    cleaned = user_id.strip()
    if len(cleaned) > 40:
        return False, "User ID exceeds 40 characters."
    return True, ""


def process_batch_dataframe(
    df: pd.DataFrame,
    text_col: str,
    user_col: Optional[str],
    predict_fn,
    network_context_fn,
    recommend_fn,
    graph,
    metrics_map
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Processes a batch DataFrame through the moderation and recommendation pipeline.
    """
    results = []
    stats = {"Total": len(df), "Allow": 0, "Review": 0, "Flag": 0}

    for idx, row in df.iterrows():
        raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        raw_user = str(row[user_col]).strip() if (user_col and user_col in row and pd.notna(row[user_col])) else ""

        if not raw_text.strip():
            continue

        clf_res = predict_fn(raw_text)
        net_ctx = network_context_fn(raw_user, graph, metrics_map) if raw_user else None
        rec = recommend_fn(clf_res, net_ctx)

        stats[rec.action] = stats.get(rec.action, 0) + 1

        results.append({
            "User ID": raw_user if raw_user else "anonymous",
            "Comment": raw_text,
            "Predicted Label": clf_res.predicted_label,
            "Confidence": round(clf_res.confidence, 3),
            "Recommended Action": rec.action,
            "Priority": rec.urgency,
            "Reason": rec.primary_reason,
            "Network Context": rec.network_context_note
        })

    return pd.DataFrame(results), stats

