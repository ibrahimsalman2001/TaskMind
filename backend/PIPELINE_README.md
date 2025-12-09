# TaskMind Multi-Modal Classification Pipeline

## Overview

This pipeline classifies YouTube videos using three independent modalities:
1. **Computer Vision (CV)** - Analyzes 6 extracted frames from the video
2. **Audio** - Transcribes audio and classifies the transcription
3. **Metadata** - Classifies title, description, and tags

Each modality produces classification scores for 22 categories, and the final result is an aggregated score from all three.

## Architecture

```
Video URL + Metadata
    │
    ├─── CV Pipeline ────┐
    │   (6 frames)       │
    │                    │
    ├─── Audio Pipeline ─┼─── Aggregation ──── Final Classification
    │   (transcribe)     │
    │                    │
    └─── Metadata ───────┘
        (title, desc, tags)
```

## Classification Categories

The pipeline classifies videos into 22 categories:

1. Academia & Explainer
2. Science & Technology
3. History & Documentaries
4. Finance & Business
5. Comedy & Skits
6. Film & Animation
7. Music & Dance
8. Gaming & Esports
9. Vlogs (General)
10. Travel & Lifestyle
11. Beauty & Fashion
12. Food & Cooking
13. News & Politics
14. Current Events Commentary
15. Health & Wellness
16. Sports (Professional)
17. Islamic/Religious
18. DIY & Craft
19. Autos & Vehicles
20. Pets & Animals
21. Kids
22. Adult/Flagged
23. Other

## Components

### 1. Computer Vision Classifier (`cv_classifier.py`)
- Extracts 6 frames from video
- Uses EfficientNet-B0 for frame classification
- Matches ImageNet labels with category keywords
- Returns normalized scores for all categories

### 2. Audio Transcriber (`audio_transcriber.py`)
- Downloads video using yt-dlp
- Extracts audio using ffmpeg
- Transcribes using OpenAI Whisper
- Chunks transcription into segments
- Classifies each chunk using SBERT + Logistic Regression
- Aggregates scores across chunks

### 3. Metadata Classifier (`metadata_classifier.py`)
- Combines title, description, and tags
- Encodes using SBERT
- Classifies using trained Logistic Regression model
- Returns probability scores for all categories

### 4. Pipeline Orchestrator (`pipeline.py`)
- Coordinates all three classifiers
- Aggregates scores with configurable weights
- Returns individual and final classifications

## Usage

### API Endpoint

```bash
POST /classify
```

**Request Body:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=...",
  "title": "Video Title",
  "description": "Video description text",
  "tags": "tag1, tag2, tag3",
  "num_frames": 6,
  "weights": {
    "cv": 0.33,
    "audio": 0.33,
    "metadata": 0.34
  }
}
```

**Response:**
```json
{
  "cv_classification": {
    "Academia & Explainer": 0.05,
    "Science & Technology": 0.12,
    ...
  },
  "audio_classification": {
    "Academia & Explainer": 0.08,
    ...
  },
  "metadata_classification": {
    "Academia & Explainer": 0.15,
    ...
  },
  "final_classification": {
    "Academia & Explainer": 0.093,
    ...
  },
  "top_category": "Science & Technology",
  "top_confidence": 0.45,
  "cv_top_5": [("Science & Technology", 0.12), ...],
  "audio_top_5": [...],
  "metadata_top_5": [...],
  "final_top_5": [...]
}
```

### Python Usage

```python
from backend.app.multimodal.pipeline import classify_video_pipeline

result = classify_video_pipeline(
    video_url="https://www.youtube.com/watch?v=...",
    title="Video Title",
    description="Description",
    tags="tag1, tag2",
    num_frames=6,
    weights={"cv": 0.33, "audio": 0.33, "metadata": 0.34}
)

print(f"Final Category: {result['top_category']}")
print(f"Confidence: {result['top_confidence']}")
```

## Running the API

```bash
cd backend/app
python main.py
```

Or with uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```

Access Swagger UI at: http://localhost:8000/docs

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for CV models
- `openai-whisper` - Audio transcription
- `sentence-transformers` - Text embeddings
- `yt-dlp` - Video downloading
- `opencv-python` - Frame extraction
- `ffmpeg-python` - Audio extraction

## Model Files Required

The pipeline expects these model files in the `models/` directory:
- `taskmind_classifier.pkl` - Trained classifier
- `label_encoder.pkl` - Label encoder
- `sbert_encoder/` - Sentence-BERT model (optional, falls back to pre-trained)

## Notes

- The CV classifier uses the existing implementation (not modified)
- Audio transcription may take time for long videos
- All three modalities run independently and can fail gracefully
- Default weights are equal (1/3 each) but can be customized
- The pipeline normalizes scores to probability distributions


