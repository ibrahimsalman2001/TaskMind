# Quick Start Guide - Manual Pipeline Execution

## Option 1: Full Manual Input (Recommended)

Run the interactive script that prompts for all inputs:

```bash
cd backend/app/multimodal
python run_pipeline_manual.py
```

This will prompt you for:
- YouTube Video URL (required)
- Video Title (optional but recommended)
- Video Description (optional, multi-line)
- Video Tags (optional, comma-separated)
- Number of frames (default: 6)
- Model weights (optional, default: equal)

## Option 2: Quick Input (Simple)

Run the simplified version that only asks for URL and title:

```bash
cd backend/app/multimodal
python run_pipeline_simple.py
```

## Option 3: Direct Python Script

Create your own script:

```python
from pipeline import classify_video_pipeline

result = classify_video_pipeline(
    video_url="https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    title="Your Video Title",
    description="Video description here",
    tags="tag1, tag2, tag3",
    num_frames=6,
    weights=None  # Equal weights, or {"cv": 0.33, "audio": 0.33, "metadata": 0.34}
)

print(f"Final Category: {result['top_category']}")
print(f"Confidence: {result['top_confidence']:.2%}")
```

## Example Usage

### Example 1: Educational Video
```python
result = classify_video_pipeline(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    title="Python Tutorial for Beginners",
    description="Learn Python programming from scratch",
    tags="python, programming, tutorial, education"
)
```

### Example 2: Gaming Video
```python
result = classify_video_pipeline(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    title="PUBG Mobile Gameplay",
    description="Epic gameplay and kills",
    tags="gaming, pubg, mobile, gameplay"
)
```

### Example 3: Music Video
```python
result = classify_video_pipeline(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    title="New Song Release",
    description="Official music video",
    tags="music, song, new release"
)
```

## Output Format

The pipeline returns a dictionary with:

```python
{
    "cv_classification": {category: score, ...},      # CV scores for all 22 categories
    "audio_classification": {category: score, ...},  # Audio scores for all 22 categories
    "metadata_classification": {category: score, ...}, # Metadata scores for all 22 categories
    "final_classification": {category: score, ...},   # Final aggregated scores
    "top_category": "Category Name",                   # Highest scoring category
    "top_confidence": 0.85,                            # Confidence score (0-1)
    "cv_top_5": [("Category", score), ...],           # Top 5 from CV
    "audio_top_5": [("Category", score), ...],        # Top 5 from Audio
    "metadata_top_5": [("Category", score), ...],     # Top 5 from Metadata
    "final_top_5": [("Category", score), ...]         # Top 5 final
}
```

## Custom Weights

You can adjust the importance of each modality:

```python
weights = {
    "cv": 0.5,      # 50% weight on CV
    "audio": 0.3,   # 30% weight on Audio
    "metadata": 0.2 # 20% weight on Metadata
}

result = classify_video_pipeline(
    video_url="...",
    title="...",
    weights=weights
)
```

## Troubleshooting

### Error: "Video URL not found"
- Make sure the URL is a valid YouTube URL
- Check your internet connection
- Ensure yt-dlp is installed: `pip install yt-dlp`

### Error: "ffmpeg not found"
- Install ffmpeg:
  - Windows: Download from https://ffmpeg.org
  - Linux: `sudo apt-get install ffmpeg`
  - Mac: `brew install ffmpeg`

### Slow Performance
- Reduce number of frames (e.g., `num_frames=3`)
- Use smaller Whisper model (edit `audio_transcriber.py`)

### No Classification Results
- Make sure keywords.json exists in the same directory
- Check that the video has audio (for audio classification)
- Verify metadata is provided (title/description/tags)

