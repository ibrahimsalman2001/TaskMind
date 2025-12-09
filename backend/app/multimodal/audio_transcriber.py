# audio_transcriber.py

import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict
import whisper
import torch

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Audio transcriber using device: {device}")

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")  # Can be "tiny", "base", "small", "medium", "large"

# Use keyword-based classification instead of trained model
# This allows us to use all 22 categories without retraining
import json
import re
from collections import defaultdict
from pathlib import Path

# Load keywords.json for keyword-based classification
keywords_path = Path(__file__).parent / "keywords.json"
with open(keywords_path, "r", encoding="utf-8") as f:
    CATEGORY_KEYWORDS = json.load(f)


def download_video(video_url: str, output_dir: str) -> str:
    """
    Download a YouTube video using yt-dlp and return the local file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete old video files to force fresh download
    for ext in ("mp4", "mkv", "webm"):
        old_file = output_dir / f"video.{ext}"
        if old_file.exists():
            old_file.unlink()

    output_template = str(output_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-o", output_template,
        video_url,
    ]

    subprocess.run(cmd, check=True)

    # Find the downloaded file
    for ext in ("mp4", "mkv", "webm", "m4a", "webm"):
        candidate = output_dir / f"video.{ext}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("Could not find downloaded video file.")


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract audio from video file and save as WAV.
    Returns path to audio file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_path = output_dir / "audio.wav"
    
    # Use ffmpeg to extract audio
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16kHz sample rate (good for Whisper)
        "-ac", "1",  # Mono
        "-y",  # Overwrite
        str(audio_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return str(audio_path)


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using Whisper.
    Returns full transcription text.
    """
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better classification.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    
    return chunks


def get_category_scores_from_text(tokens: list) -> Dict[str, int]:
    """
    Given a list of words from transcription, return raw match scores per category.
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


def classify_text_chunks(chunks: List[str]) -> Dict[str, float]:
    """
    Classify text chunks using keyword matching and return aggregated scores.
    """
    if not chunks:
        return {cat: 0.0 for cat in CATEGORY_KEYWORDS.keys()}
    
    # Combine all chunks
    full_text = " ".join(chunks)
    
    # Extract tokens (words) from text
    tokens = re.findall(r"\w+", full_text.lower())
    
    # Get raw scores
    raw_scores = get_category_scores_from_text(tokens)
    
    # Normalize scores
    norm_scores = normalize_scores(raw_scores)
    
    return norm_scores


def classify_video_audio(video_url: str) -> Dict[str, float]:
    """
    Main function: Download video, extract audio, transcribe, chunk, and classify.
    Returns category scores dictionary.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="taskmind_audio_") as tmp_dir:
            # Download video
            print(f"Downloading video from {video_url}...")
            video_path = download_video(video_url, tmp_dir)
            
            # Extract audio
            print("Extracting audio...")
            audio_path = extract_audio(video_path, tmp_dir)
            
            # Transcribe
            print("Transcribing audio...")
            transcription = transcribe_audio(audio_path)
            print(f"Transcription length: {len(transcription)} characters")
            
            if not transcription or len(transcription.strip()) == 0:
                print("Warning: Empty transcription, returning empty scores")
                return {}
            
            # Chunk text
            print("Chunking transcription...")
            chunks = chunk_text(transcription)
            print(f"Created {len(chunks)} chunks")
            
            if not chunks:
                print("Warning: No chunks created, returning empty scores")
                return {}
            
            # Classify chunks
            print("Classifying audio transcription...")
            scores = classify_text_chunks(chunks)
            
            return scores
    except Exception as e:
        print(f"Error in audio classification: {e}")
        raise

