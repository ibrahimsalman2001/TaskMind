# cv_classifier.py

import re
from typing import List, Dict
from collections import defaultdict
import json
from vision_model import classify_frame
# (Later: from ocr_module import extract_text_from_frame)

# Load keywords.json
with open("app/multimodal/keywords.json", "r", encoding="utf-8") as f:
    CATEGORY_KEYWORDS = json.load(f)


def get_category_scores_from_text(tokens: List[str]) -> Dict[str, int]:
    """
    Given a list of words from visual/OCR labels, return raw match scores per category.
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


def classify_video_by_cv(frames: List[str]) -> Dict[str, float]:
    """
    Main CV classifier function.
    Takes a list of video frame paths, runs EfficientNet on each,
    and returns a normalized category score vector.
    """
    all_visual_labels: List[str] = []

    for frame_path in frames:
        top_predictions = classify_frame(frame_path, top_k=3)
        labels_only = [label for label, _ in top_predictions]
        all_visual_labels.extend(labels_only)

    # TODO: Integrate real OCR here
    fake_ocr_text = "PUBG kills 20 livestream bayan prophet Muhammad naat"

    ocr_tokens = re.findall(r"\w+", fake_ocr_text.lower())
    all_tokens = all_visual_labels + ocr_tokens

    raw_scores = get_category_scores_from_text(all_tokens)
    norm_scores = normalize_scores(raw_scores)

    return norm_scores