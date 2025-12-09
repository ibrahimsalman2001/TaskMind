"""
Core pipeline aggregation and scoring logic.

This module contains the aggregation functions used to combine scores
from all three pipelines (CV, Audio, Metadata) into a final classification.
"""

from typing import Dict, Optional
import json
from pathlib import Path


# Load category list from keywords.json to ensure all categories are present
keywords_path = Path(__file__).parent / "keywords.json"
with open(keywords_path, "r", encoding="utf-8") as f:
    CATEGORIES = list(json.load(f).keys())


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to sum to 1.0 (probability distribution).
    
    Args:
        scores: Dictionary of category scores
    
    Returns:
        Normalized scores summing to 1.0
    """
    total = sum(scores.values())
    if total == 0:
        # Return uniform distribution if all scores are zero
        return {cat: 1.0 / len(CATEGORIES) for cat in CATEGORIES}
    return {cat: score / total for cat, score in scores.items()}


def aggregate_scores(
    cv_scores: Dict[str, float],
    audio_scores: Dict[str, float],
    metadata_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Aggregate scores from all three pipelines with optional weighting.
    
    Args:
        cv_scores: Computer vision classification scores
        audio_scores: Audio transcription classification scores
        metadata_scores: Metadata classification scores
        weights: Optional weights for each pipeline (default: equal weights of 1/3 each)
                 Format: {"cv": 0.33, "audio": 0.33, "metadata": 0.34}
    
    Returns:
        Final aggregated scores for each category, normalized to sum to 1.0
    """
    if weights is None:
        weights = {"cv": 1.0/3, "audio": 1.0/3, "metadata": 1.0/3}
    
    # Normalize each set of scores
    cv_norm = normalize_scores(cv_scores)
    audio_norm = normalize_scores(audio_scores)
    metadata_norm = normalize_scores(metadata_scores)
    
    # Initialize final scores
    final_scores = {cat: 0.0 for cat in CATEGORIES}
    
    # Weighted aggregation
    for category in CATEGORIES:
        final_scores[category] = (
            weights["cv"] * cv_norm.get(category, 0.0) +
            weights["audio"] * audio_norm.get(category, 0.0) +
            weights["metadata"] * metadata_norm.get(category, 0.0)
        )
    
    # Normalize final scores
    final_scores = normalize_scores(final_scores)
    
    return final_scores
