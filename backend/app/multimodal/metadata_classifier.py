# metadata_classifier.py

import re
import json
from typing import Dict
from collections import defaultdict
from pathlib import Path

# Load keywords.json for keyword-based classification
keywords_path = Path(__file__).parent / "keywords.json"
with open(keywords_path, "r", encoding="utf-8") as f:
    CATEGORY_KEYWORDS = json.load(f)


def get_category_scores_from_text(tokens: list) -> Dict[str, int]:
    """
    Given a list of words from metadata, return raw match scores per category.
    """
    scores = defaultdict(int)
    token_set = set(t.lower() for t in tokens)
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in token_set:
                scores[category] += 1
    
    return scores


def normalize_scores(score_dict: Dict[str, int]) -> Dict[str, float]:
    """
    Normalize category scores to sum to 1.
    """
    total = sum(score_dict.values())
    if total == 0:
        return {cat: 0.0 for cat in CATEGORY_KEYWORDS.keys()}
    return {cat: round(score / total, 4) for cat, score in score_dict.items()}


def classify_metadata(title: str, description: str = "", tags: str = "") -> Dict[str, float]:
    """
    Classify video metadata (title, description, tags) using keyword matching.
    Returns category scores for all 22 categories.
    
    Args:
        title: Video title
        description: Video description (optional)
        tags: Video tags (optional)
    
    Returns:
        Dictionary mapping category names to confidence scores
    """
    # Combine metadata into single text
    text = f"{title} {description} {tags}".strip()
    
    if not text:
        return {cat: 0.0 for cat in CATEGORY_KEYWORDS.keys()}
    
    # Extract tokens (words) from text
    tokens = re.findall(r"\w+", text.lower())
    
    # Get raw scores
    raw_scores = get_category_scores_from_text(tokens)
    
    # Normalize scores
    norm_scores = normalize_scores(raw_scores)
    
    return norm_scores

