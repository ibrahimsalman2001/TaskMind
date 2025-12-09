# pipeline.py

from typing import Dict, List, Optional
import json
import os
from pathlib import Path
from cv_classifier import classify_video_by_cv
from frame_extractor import download_and_extract_frames
from metadata_classifier import classify_metadata
from audio_transcriber import classify_video_audio

# Load category list from keywords.json to ensure all categories are present
keywords_path = Path(__file__).parent / "keywords.json"
with open(keywords_path, "r", encoding="utf-8") as f:
    CATEGORIES = list(json.load(f).keys())


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to sum to 1.0 (probability distribution).
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
    Aggregate scores from all three models with optional weighting.
    
    Args:
        cv_scores: Computer vision classification scores
        audio_scores: Audio transcription classification scores
        metadata_scores: Metadata classification scores
        weights: Optional weights for each model (default: equal weights)
                 Format: {"cv": 0.33, "audio": 0.33, "metadata": 0.34}
    
    Returns:
        Final aggregated scores for each category
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


def get_top_category(scores: Dict[str, float], top_k: int = 1) -> List[tuple]:
    """
    Get top K categories by score.
    
    Returns:
        List of (category, score) tuples, sorted by score descending
    """
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_k]


def classify_video_pipeline(
    video_url: str,
    title: str,
    description: str = "",
    tags: str = "",
    num_frames: int = 6,
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Main pipeline function that classifies a video using all three modalities.
    
    Args:
        video_url: YouTube video URL
        title: Video title
        description: Video description (optional)
        tags: Video tags (optional)
        num_frames: Number of frames to extract for CV (default: 6)
        weights: Optional weights for each model
    
    Returns:
        Dictionary containing:
        - cv_classification: CV model scores
        - audio_classification: Audio model scores
        - metadata_classification: Metadata model scores
        - final_classification: Aggregated scores
        - top_category: Category with highest final score
        - top_confidence: Confidence score for top category
    """
    print("=" * 60)
    print("Starting TaskMind Multi-Modal Classification Pipeline")
    print("=" * 60)
    
    results = {}
    
    # 1. Computer Vision Classification
    print("\n[1/3] Computer Vision Classification...")
    try:
        frames = download_and_extract_frames(video_url, num_frames=num_frames)
        cv_scores = classify_video_by_cv(frames)
        results["cv_classification"] = cv_scores
        print(f"✓ CV classification complete. Top category: {get_top_category(cv_scores, 1)[0][0]}")
    except Exception as e:
        print(f"✗ CV classification failed: {e}")
        cv_scores = {cat: 0.0 for cat in CATEGORIES}
        results["cv_classification"] = cv_scores
    
    # 2. Audio Transcription & Classification
    print("\n[2/3] Audio Transcription & Classification...")
    try:
        audio_scores = classify_video_audio(video_url)
        results["audio_classification"] = audio_scores
        print(f"✓ Audio classification complete. Top category: {get_top_category(audio_scores, 1)[0][0]}")
    except Exception as e:
        print(f"✗ Audio classification failed: {e}")
        audio_scores = {cat: 0.0 for cat in CATEGORIES}
        results["audio_classification"] = audio_scores
    
    # 3. Metadata Classification
    print("\n[3/3] Metadata Classification...")
    try:
        metadata_scores = classify_metadata(title, description, tags)
        results["metadata_classification"] = metadata_scores
        print(f"✓ Metadata classification complete. Top category: {get_top_category(metadata_scores, 1)[0][0]}")
    except Exception as e:
        print(f"✗ Metadata classification failed: {e}")
        metadata_scores = {cat: 0.0 for cat in CATEGORIES}
        results["metadata_classification"] = metadata_scores
    
    # 4. Final Aggregation
    print("\n[Aggregation] Combining all classifications...")
    final_scores = aggregate_scores(cv_scores, audio_scores, metadata_scores, weights)
    results["final_classification"] = final_scores
    
    # Get top category
    top_category, top_confidence = get_top_category(final_scores, 1)[0]
    results["top_category"] = top_category
    results["top_confidence"] = round(top_confidence, 4)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT: {top_category} (confidence: {top_confidence:.2%})")
    print(f"{'=' * 60}\n")
    
    # Add top 5 categories for each modality
    results["cv_top_5"] = get_top_category(cv_scores, 5)
    results["audio_top_5"] = get_top_category(audio_scores, 5)
    results["metadata_top_5"] = get_top_category(metadata_scores, 5)
    results["final_top_5"] = get_top_category(final_scores, 5)
    
    return results


if __name__ == "__main__":
    # Example usage
    test_url = "https://www.youtube.com/watch?v=47fLXANW39k"
    test_title = "Example Video Title"
    test_description = "This is an example video description"
    test_tags = "example, test, video"
    
    result = classify_video_pipeline(
        video_url=test_url,
        title=test_title,
        description=test_description,
        tags=test_tags,
        num_frames=6
    )
    
    print("\nDetailed Results:")
    print(json.dumps(result, indent=2, default=str))

