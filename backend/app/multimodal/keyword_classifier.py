# keyword_classifier.py

import json
import re
from typing import Dict, Tuple
from pathlib import Path


# Load your full keyword list from the JSON file once
keywords_path = Path(__file__).parent / "keywords.json"
with open(keywords_path, "r", encoding="utf-8") as f:
    CATEGORY_KEYWORDS: Dict[str, list] = json.load(f)


def keyword_based_classifier(text: str) -> Tuple[str, int]:
    """
    Classifies the input text (from OCR + metadata) into one of the TaskMind categories
    based on keyword presence.

    Returns:
        - category name
        - number of matching keywords (for debug)
    """
    text = text.lower()
    category_match_count: Dict[str, int] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        match_count = 0
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text):
                match_count += 1
        if match_count > 0:
            category_match_count[category] = match_count

    if category_match_count:
        # Return category with most matched keywords
        best_category = max(category_match_count, key=category_match_count.get)
        return best_category, category_match_count[best_category]

    return "Other", 0