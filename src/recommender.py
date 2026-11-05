"""
Recommendation Logic Module
---------------------------
Provides transparent, rule-based moderation recommendations combining:
1. Classifier outputs (predicted label & confidence)
2. Network context features (centrality, degree, community)

ETHICAL / SYSTEM PRINCIPLE:
Network metrics provide structural context regarding reach and amplification potential.
They must NEVER be treated as proof of cyberbullying or guilt. Decisions must remain
transparent, explainable, and subject to final human moderator review.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.classifier import ClassifierResult


@dataclass
class ModerationRecommendation:
    action: str  # "Allow", "Review", "Flag"
    urgency: str  # "Normal", "Moderate", "High"
    primary_reason: str
    network_context_note: str
    combined_explanation: str


def generate_recommendation(
    classifier_result: ClassifierResult,
    user_network_context: Optional[Dict[str, Any]] = None
) -> ModerationRecommendation:
    """
    Evaluates classifier output and optional user network metrics to form a transparent
    action recommendation for human moderators.
    
    Args:
        classifier_result: Classification predictions (label, confidence).
        user_network_context: Optional dictionary with user degree, centrality, etc.
        
    Returns:
        ModerationRecommendation with action (Allow/Review/Flag) and clear rationale.
    """
    label = classifier_result.predicted_label
    conf = classifier_result.confidence

    # Extract network context safely
    has_network_data = user_network_context is not None and user_network_context.get("is_known_in_graph", False)
    betweenness = user_network_context.get("betweenness_centrality", 0.0) if has_network_data else 0.0
    degree = user_network_context.get("degree", 0) if has_network_data else 0

    # Baseline network context note
    if has_network_data:
        if betweenness > 0.20:
            network_note = (
                f"User acts as a network bridge (Betweenness: {betweenness:.2f}, Connections: {degree}). "
                "High broadcast reach increases potential audience exposure if content is abusive, "
                "serving strictly as structural context."
            )
        elif degree > 5:
            network_note = (
                f"User has high local connectivity ({degree} direct connections). "
                "Amplification factor is elevated within local community."
            )
        else:
            network_note = (
                f"User has moderate network presence ({degree} connections, Betweenness: {betweenness:.2f}). "
                "No unusual broadcast reach detected."
            )
    else:
        network_note = "No prior interaction graph context available for this user identifier."

    # Rule evaluation
    # Rule 1: High-confidence harmful content -> Flag
    if label in ["Toxic", "Harassment"] and conf >= 0.80:
        action = "Flag"
        urgency = "High" if betweenness > 0.15 else "Moderate"
        primary_reason = f"High confidence ({conf * 100:.1f}%) detection of {label.lower()} content."
        combined_explanation = (
            f"Automated classifier detected high-confidence {label.lower()} text ({conf * 100:.1f}%). "
            f"Recommended for immediate moderator confirmation. {network_note}"
        )
        return ModerationRecommendation(
            action=action,
            urgency=urgency,
            primary_reason=primary_reason,
            network_context_note=network_note,
            combined_explanation=combined_explanation
        )

    # Rule 2: Moderate confidence harmful content OR borderline ambiguity -> Review
    if (label in ["Toxic", "Harassment"] and conf < 0.80) or (label == "Clean" and conf < 0.70):
        action = "Review"
        urgency = "Moderate" if betweenness > 0.20 else "Normal"
        primary_reason = f"Borderline confidence ({conf * 100:.1f}%) for label '{label}'. Human review advised."
        combined_explanation = (
            f"Classifier confidence ({conf * 100:.1f}%) falls in the ambiguous threshold. "
            f"Requires moderator judgment to assess nuance or potential false positive. {network_note}"
        )
        return ModerationRecommendation(
            action=action,
            urgency=urgency,
            primary_reason=primary_reason,
            network_context_note=network_note,
            combined_explanation=combined_explanation
        )

    # Rule 3: Confident Clean content -> Allow
    action = "Allow"
    urgency = "Normal"
    primary_reason = f"Content classified as clean ({conf * 100:.1f}% confidence)."
    combined_explanation = (
        f"No toxic or harassing language detected. Content conforms to safety baselines. "
        f"{network_note}"
    )
    return ModerationRecommendation(
        action=action,
        urgency=urgency,
        primary_reason=primary_reason,
        network_context_note=network_note,
        combined_explanation=combined_explanation
    )
