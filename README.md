# TaskMind - Multi-Modal Video Classification Pipeline

A comprehensive Python-based system for classifying YouTube videos using three complementary AI pipelines: Computer Vision, Audio Transcription, and Metadata Analysis.

## Overview

TaskMind analyzes videos from multiple modalities and combines the results using intelligent aggregation to produce highly accurate content classifications across 22 categories.

### Three Main Pipelines

#### 1. **Computer Vision Pipeline**
- Extracts keyframes from the video at regular intervals
- Classifies each frame using EfficientNet-B0 (ImageNet pre-trained)
- Performs OCR (Optical Character Recognition) to detect text in frames
- Produces frame-level predictions and visual content scores

#### 2. **Audio Transcription Pipeline**
- Extracts and converts video audio to WAV format
- Transcribes audio using OpenAI Whisper (automatic speech recognition)
- Analyzes transcription text for content keywords
- Produces audio-based category scores

#### 3. **Metadata Classification Pipeline**
- Analyzes video title, description, and tags
- Matches content against keyword taxonomy
- Produces metadata-based category scores

### Final Classification

Results from all three pipelines are normalized and aggregated using equal weighting to produce a final confidence score for each of the 22 content categories.

## Categories Supported

The system classifies content into 22 categories:

- Academia & Explainer
- Science & Technology
- History & Documentaries
- Finance & Business
- Comedy & Skits
- Film & Animation
- Music & Dance
- Gaming & Esports
- Vlogs (General)
- Travel & Lifestyle
- Beauty & Fashion
- Food & Cooking
- News & Politics
- Current Events Commentary
- Health & Wellness
- Sports (Professional)
- Islamic/Religious
- DIY & Craft
- Autos & Vehicles
- Pets & Animals
- Kids
- Adult/Flagged
- Other (fallback / low-confidence / ambiguous content)

## Project Structure

```
backend/app/multimodal/
├── main.py                      # Main entry point (interactive & CLI modes)
├── pipeline.py                  # Core aggregation and scoring logic
├── frame_extractor.py           # Video download and frame extraction
├── vision_model.py              # EfficientNet-B0 frame classification
├── ocr_module.py                # EasyOCR text extraction
├── audio_transcriber.py         # Whisper-based transcription
├── metadata_classifier.py       # Keyword-based metadata analysis
├── keywords.json                # Category keyword definitions
└── classification_result.json   # Output results file
```

## Installation

### Requirements

- Python 3.8+
- FFmpeg (for video/audio processing)
- PyTorch with CUDA support (optional, falls back to CPU)

### Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure FFmpeg is installed:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (with conda)
conda install ffmpeg
```

3. (Optional) For restricted YouTube videos, create a `cookies.txt` file in the backend/app/multimodal directory using yt-dlp:

```bash
yt-dlp --cookies-from-browser firefox --save-cookies cookies.txt https://www.youtube.com/
```

## Usage

### Interactive Mode

Run without arguments to enter interactive mode where you'll be prompted for inputs:

```bash
cd backend/app/multimodal
python main.py
```

This will guide you through:
1. Choosing between YouTube URL or local video file
2. Entering video title, description, and tags
3. Specifying authentication cookies (if needed)
4. Choosing number of frames to analyze

### Non-Interactive Mode (CLI)

Provide all arguments directly:

```bash
cd backend/app/multimodal

# From YouTube URL
python main.py --url "https://www.youtube.com/watch?v=..." \
               --title "Video Title" \
               --description "Video description" \
               --tags "tag1,tag2"

# From local video file (no download required)
python main.py --input-file /path/to/video.mp4 \
               --title "Video Title"

# With authentication cookies
python main.py --url "https://www.youtube.com/watch?v=..." \
               --title "Video Title" \
               --cookies /path/to/cookies.txt

# Custom frame count
python main.py --input-file video.mp4 \
               --title "Title" \
               --frames 12
```

### Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--url` | string | No | YouTube video URL |
| `--title` | string | Yes* | Video title |
| `--description` | string | No | Video description |
| `--tags` | string | No | Comma-separated tags |
| `--frames` | int | No | Number of frames to extract (default: 6) |
| `--input-file` | string | No | Path to local video file (overrides --url) |
| `--cookies` | string | No | Path to cookies.txt for authentication |

*Either `--url` or `--input-file` is required

## Output

The pipeline generates a detailed JSON file (`classification_result.json`) containing:

```json
{
  "cv_predictions": {
    "per_frame_predictions": [...],
    "total_frames_analyzed": 6
  },
  "ocr_text": "extracted text from frames...",
  "cv_scores": {
    "category1": 0.25,
    "category2": 0.18
  },
  "transcription": "full audio transcription...",
  "audio_scores": {
    "category1": 0.30,
    "category2": 0.15
  },
  "metadata_scores": {
    "category1": 0.22,
    "category2": 0.20
  },
  "final_scores": {
    "category1": 0.25,
    "category2": 0.18
  },
  "top_category": {
    "category": "Film & Animation",
    "confidence": 0.3421
  }
}
```

## Performance Characteristics

- **Frame Extraction**: ~10-30 seconds (depends on video length)
- **Frame Classification**: ~1-2 seconds per frame
- **OCR**: ~5-15 seconds for multiple frames
- **Audio Transcription**: ~1-5 minutes (depends on audio length and hardware)
- **Total Pipeline**: ~10-20 minutes on CPU, ~2-5 minutes on GPU

## Troubleshooting

### YouTube Download Issues

If videos fail to download:

1. **Check FFmpeg installation**: `ffmpeg -version`
2. **Try with cookies**: Download cookies and use `--cookies` argument
3. **Use local file**: If download fails persistently, download manually and use `--input-file`

### Audio Transcription is Slow

The Whisper model runs on CPU by default, which is slow. To use GPU:

```python
# Modify audio_transcriber.py line 1:
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Then install CUDA-enabled PyTorch if not already installed.

### Memory Issues

If experiencing memory issues with large videos:

1. Reduce `--frames` to process fewer frames
2. Use a smaller video or pre-clip to a shorter segment
3. Close other applications

## File Descriptions

### Core Pipeline Files

- **main.py**: Entry point with interactive and CLI modes
- **pipeline.py**: Core aggregation logic and scoring
- **keywords.json**: Category definitions and keywords for matching

### Modality Pipelines

- **frame_extractor.py**: Downloads videos and extracts keyframes
- **vision_model.py**: Frame classification using EfficientNet-B0
- **ocr_module.py**: Text extraction from frames using EasyOCR
- **audio_transcriber.py**: Audio extraction and Whisper transcription
- **metadata_classifier.py**: Title/description/tags classification

## Dependencies

Key Python packages used:

- `torch`, `torchvision`: Deep learning framework and pre-trained models
- `opencv-python`: Video processing
- `openai-whisper`: Audio transcription
- `easyocr`: Optical character recognition
- `requests`: HTTP requests
- `scikit-learn`: Machine learning utilities

See `requirements.txt` for complete list and versions.

## Development

The codebase is organized for clarity and maintainability:

- Each pipeline is independently testable
- All imports are at the top of files
- Error handling is graceful with fallbacks
- Results are saved to JSON for further analysis
- Logging provides visibility into each step

### Adding New Categories

To add new categories:

1. Update `keywords.json` with new category and keywords
2. The system automatically adapts—no code changes needed

## License

This project is provided for educational and research purposes.

## Support

For issues or questions:

1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Review output logs for specific error messages
4. Ensure input video is accessible and valid format

---

Last updated: December 2025
