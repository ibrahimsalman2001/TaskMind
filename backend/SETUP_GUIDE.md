# TaskMind Pipeline Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** You also need `ffmpeg` installed on your system:
- **Windows**: Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`
- **Mac**: `brew install ffmpeg`

### 2. Verify Model Files

Ensure these files exist in `TaskMind/models/`:
- `taskmind_classifier.pkl`
- `label_encoder.pkl`
- `sbert_encoder/` (optional - will use pre-trained if missing)

### 3. Run the API

```bash
cd backend/app
python main.py
```

Or:
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test the API

Open http://localhost:8000/docs for Swagger UI

Example request:
```json
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "title": "Example Video",
  "description": "This is an example",
  "tags": "example, test"
}
```

## Pipeline Flow

1. **Input**: Video URL + Metadata (title, description, tags)

2. **CV Processing**:
   - Downloads video
   - Extracts 6 frames
   - Classifies each frame with EfficientNet
   - Matches labels with category keywords
   - Returns normalized scores

3. **Audio Processing**:
   - Downloads video (reuses if already downloaded)
   - Extracts audio to WAV
   - Transcribes with Whisper
   - Chunks transcription
   - Classifies each chunk
   - Aggregates scores

4. **Metadata Processing**:
   - Combines title + description + tags
   - Encodes with SBERT
   - Classifies with trained model
   - Returns scores

5. **Aggregation**:
   - Normalizes all three score sets
   - Applies weights (default: equal)
   - Combines scores
   - Returns final classification

## Configuration

### Custom Weights

You can adjust the importance of each modality:

```python
weights = {
    "cv": 0.5,      # 50% weight on CV
    "audio": 0.3,   # 30% weight on Audio
    "metadata": 0.2 # 20% weight on Metadata
}
```

### Number of Frames

Adjust `num_frames` parameter (default: 6):
- More frames = better CV accuracy but slower
- Recommended: 6-10 frames

### Whisper Model Size

In `audio_transcriber.py`, change:
```python
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
```

- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `large`: Most accurate, slowest

## Troubleshooting

### Error: "ffmpeg not found"
- Install ffmpeg system-wide
- Ensure it's in your PATH

### Error: "Model files not found"
- Check that `models/` directory exists
- Verify `taskmind_classifier.pkl` and `label_encoder.pkl` are present

### Error: "CUDA out of memory"
- Reduce batch sizes in classification
- Use CPU instead: Set `device = "cpu"` in modules
- Use smaller Whisper model

### Slow Performance
- Use smaller Whisper model (`tiny` or `base`)
- Reduce number of frames
- Use GPU if available

## File Structure

```
TaskMind/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application
│   │   └── multimodal/
│   │       ├── pipeline.py            # Main orchestrator
│   │       ├── cv_classifier.py       # CV classification
│   │       ├── audio_transcriber.py   # Audio processing
│   │       ├── metadata_classifier.py # Metadata classification
│   │       ├── frame_extractor.py     # Frame extraction (existing)
│   │       ├── vision_model.py        # Vision model (existing)
│   │       └── keywords.json          # Category keywords
│   └── imagenet_classes.txt
├── models/
│   ├── taskmind_classifier.pkl
│   ├── label_encoder.pkl
│   └── sbert_encoder/ (optional)
└── requirements.txt
```

## API Endpoints

- `GET /` - API information
- `POST /classify` - Main classification endpoint
- `GET /health` - Health check
- `GET /docs` - Swagger documentation

## Next Steps

1. Train models on your specific dataset
2. Fine-tune category keywords in `keywords.json`
3. Adjust aggregation weights based on validation
4. Add caching for repeated video classifications
5. Implement batch processing endpoint


