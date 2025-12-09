#!/usr/bin/env python3
"""
TaskMind Multi-Modal Video Classification - Main Entry Point

This script provides both interactive and non-interactive modes for classifying
YouTube videos using three complementary pipelines:

1. Computer Vision Pipeline: Extracts frames and uses EfficientNet-B0 for classification
2. Audio Transcription Pipeline: Transcribes video with Whisper and classifies content
3. Metadata Classification Pipeline: Analyzes title, description, and tags

The outputs from all three pipelines are aggregated to produce a final classification.

Usage:
    Interactive mode (prompts for inputs):
        python main.py

    Non-interactive mode (use arguments):
        python main.py --url <video_url> --title <title> --input-file <local_video_path>

    With cookies (for restricted videos):
        python main.py --url <video_url> --title <title> --cookies /path/to/cookies.txt

    Using local video file (no download):
        python main.py --input-file /path/to/video.mp4 --title "Video Title"
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict

# Import pipeline components
from frame_extractor import download_video, extract_sample_frames
from vision_model import classify_frame
from ocr_module import extract_text_from_frames
from audio_transcriber import extract_audio, transcribe_audio, chunk_text, classify_text_chunks
import metadata_classifier as mc
from pipeline import aggregate_scores, CATEGORIES


def run_pipeline(
    video_url: Optional[str] = None,
    title: Optional[str] = None,
    description: str = "",
    tags: str = "",
    num_frames: int = 6,
    input_file: Optional[str] = None,
    cookies: Optional[str] = None,
) -> Dict:
    """
    Execute the complete multi-modal classification pipeline.
    
    Args:
        video_url: YouTube video URL (required if input_file not provided)
        title: Video title (required)
        description: Video description (optional)
        tags: Video tags (optional)
        num_frames: Number of frames to extract (default: 6)
        input_file: Path to local video file (optional, bypasses download)
        cookies: Path to cookies file for authentication (optional)
    
    Returns:
        Dictionary containing results from all pipelines and final classification
    """
    if not title:
        raise ValueError("Video title is required")
    
    if not input_file and not video_url:
        raise ValueError("Either --input-file or --url is required")
    
    results = {}

    with tempfile.TemporaryDirectory(prefix="taskmind_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # === STEP 1: VIDEO SOURCE ===
        print("\n" + "=" * 70)
        print("STEP 1: VIDEO SOURCE")
        print("=" * 70)
        
        if input_file:
            video_path = input_file
            print(f"✓ Using local video file: {video_path}")
        else:
            print(f"Downloading video from: {video_url}")
            video_path = download_video(
                video_url,
                str(tmp_path),
                use_ranged=False,
                cookies_path=cookies
            )
            print(f"✓ Downloaded to: {video_path}")

        # === STEP 2: FRAME EXTRACTION & CV CLASSIFICATION ===
        print("\n" + "=" * 70)
        print("STEP 2: COMPUTER VISION PIPELINE")
        print("=" * 70)
        
        print(f"Extracting {num_frames} frames...")
        frames = extract_sample_frames(
            video_path,
            num_frames=num_frames,
            output_dir=str(tmp_path / "frames")
        )
        print(f"✓ Extracted {len(frames)} frames")

        print("\nClassifying frames...")
        per_frame_preds = []
        visual_labels = []
        
        for i, frame_path in enumerate(frames, 1):
            preds = classify_frame(frame_path, top_k=3)
            per_frame_preds.append({
                "frame_number": i,
                "predictions": preds
            })
            
            labels_only = [label for label, _ in preds]
            visual_labels.extend(labels_only)
            
            print(f"  Frame {i}: {preds[0][0]} ({preds[0][1]:.2%})")

        results["cv_predictions"] = {
            "per_frame_predictions": per_frame_preds,
            "total_frames_analyzed": len(frames)
        }

        # === STEP 3: OCR ===
        print("\nExtracting text from frames (OCR)...")
        ocr_text = extract_text_from_frames(frames)
        results["ocr_text"] = ocr_text
        print(f"✓ OCR complete ({len(ocr_text)} characters extracted)")

        # === STEP 4: CV CATEGORY SCORES ===
        print("\nComputing CV category scores...")
        import re
        ocr_tokens = re.findall(r"\w+", ocr_text.lower())
        combined_tokens = [t.lower() for t in visual_labels] + ocr_tokens
        
        try:
            from metadata_classifier import (
                get_category_scores_from_text,
                normalize_scores
            )
            raw_cv_scores = get_category_scores_from_text(combined_tokens)
            cv_scores = normalize_scores(raw_cv_scores)
        except Exception:
            cv_scores = {cat: 0.0 for cat in CATEGORIES}
        
        results["cv_scores"] = cv_scores
        top_cv_cat = max(cv_scores.items(), key=lambda x: x[1])
        print(f"✓ Top CV category: {top_cv_cat[0]} ({top_cv_cat[1]:.2%})")

        # === STEP 5: AUDIO PIPELINE ===
        print("\n" + "=" * 70)
        print("STEP 3: AUDIO TRANSCRIPTION PIPELINE")
        print("=" * 70)
        
        print("Extracting audio...")
        audio_path = extract_audio(video_path, str(tmp_path / "audio"))
        print(f"✓ Audio extracted to: {audio_path}")

        print("Transcribing audio (this may take a few minutes)...")
        transcription = transcribe_audio(audio_path)
        results["transcription"] = transcription
        
        trans_preview = transcription[:200] + "..." if len(transcription) > 200 else transcription
        print(f"✓ Transcription complete\n  Preview: {trans_preview}")

        print("\nClassifying audio content...")
        chunks = chunk_text(transcription)
        audio_scores = classify_text_chunks(chunks)
        results["audio_scores"] = audio_scores
        
        top_audio_cat = max(audio_scores.items(), key=lambda x: x[1])
        print(f"✓ Top audio category: {top_audio_cat[0]} ({top_audio_cat[1]:.2%})")

        # === STEP 6: METADATA PIPELINE ===
        print("\n" + "=" * 70)
        print("STEP 4: METADATA CLASSIFICATION PIPELINE")
        print("=" * 70)
        
        print(f"Analyzing metadata:")
        print(f"  Title: {title}")
        print(f"  Description: {description[:100]}..." if len(description) > 100 else f"  Description: {description}")
        print(f"  Tags: {tags if tags else '(none)'}")
        
        metadata_scores = mc.classify_metadata(title, description, tags)
        results["metadata_scores"] = metadata_scores
        
        top_meta_cat = max(metadata_scores.items(), key=lambda x: x[1])
        print(f"✓ Top metadata category: {top_meta_cat[0]} ({top_meta_cat[1]:.2%})")

        # === STEP 7: AGGREGATION ===
        print("\n" + "=" * 70)
        print("STEP 5: FINAL AGGREGATION & SCORING")
        print("=" * 70)
        
        print("Aggregating scores from all pipelines...")
        final_scores = aggregate_scores(cv_scores, audio_scores, metadata_scores)
        results["final_scores"] = final_scores

        # Get top category
        top_cat, top_conf = max(final_scores.items(), key=lambda x: x[1])
        results["top_category"] = {
            "category": top_cat,
            "confidence": round(top_conf, 4)
        }

        # === SUMMARY ===
        print("\n" + "=" * 70)
        print("CLASSIFICATION SUMMARY")
        print("=" * 70)
        
        print(f"\nFinal Classification: {top_cat}")
        print(f"Confidence Score: {top_conf:.2%}")
        
        print(f"\nTop 3 by pipeline:")
        print("\n  Computer Vision:")
        for cat, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    • {cat}: {score:.2%}")
        
        print("\n  Audio Transcription:")
        for cat, score in sorted(audio_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    • {cat}: {score:.2%}")
        
        print("\n  Metadata:")
        for cat, score in sorted(metadata_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    • {cat}: {score:.2%}")
        
        print("\n  Final (Aggregated):")
        for cat, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    • {cat}: {score:.2%}")

        # === SAVE RESULTS ===
        out_path = Path(__file__).parent / "classification_result.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {out_path}")
        print("=" * 70 + "\n")

        return results


def interactive_mode() -> Dict:
    """Run the pipeline in interactive mode, prompting for user inputs."""
    print("\n" + "=" * 70)
    print("TaskMind Multi-Modal Video Classification - Interactive Mode")
    print("=" * 70)
    
    print("\n--- Video Source ---")
    print("Choose input method:")
    print("  1. Download from YouTube URL")
    print("  2. Use local video file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        input_file = input("Enter path to local video file: ").strip()
        video_url = None
    else:
        video_url = input("Enter YouTube video URL: ").strip()
        input_file = None
    
    print("\n--- Metadata ---")
    title = input("Enter video title: ").strip()
    
    description = input("Enter video description (or press Enter to skip): ").strip()
    
    tags = input("Enter video tags, comma-separated (or press Enter to skip): ").strip()
    
    cookies_path = input(
        "Path to cookies.txt for authentication (or press Enter to skip): "
    ).strip()
    cookies = cookies_path if cookies_path else None
    
    num_frames = input("Number of frames to analyze (default 6): ").strip()
    num_frames = int(num_frames) if num_frames.isdigit() else 6
    
    return run_pipeline(
        video_url=video_url,
        title=title,
        description=description,
        tags=tags,
        num_frames=num_frames,
        input_file=input_file,
        cookies=cookies,
    )


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="TaskMind Multi-Modal Video Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Non-interactive with YouTube URL
  python main.py --url "https://www.youtube.com/watch?v=..." --title "Video Title"
  
  # Use local video file (no download)
  python main.py --input-file /path/to/video.mp4 --title "Video Title"
  
  # With cookies for authentication
  python main.py --url "https://www.youtube.com/watch?v=..." \\
                 --title "Video Title" \\
                 --cookies /path/to/cookies.txt
        """
    )
    
    parser.add_argument(
        "--url",
        type=str,
        help="YouTube video URL"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        help="Video title"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Video description (optional)"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Video tags, comma-separated (optional)"
    )
    
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of frames to extract (default: 6)"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to local video file (bypasses download)"
    )
    
    parser.add_argument(
        "--cookies",
        type=str,
        help="Path to cookies.txt file for authentication"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided or only help requested, use interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        # Non-interactive mode with arguments
        try:
            run_pipeline(
                video_url=args.url,
                title=args.title,
                description=args.description,
                tags=args.tags,
                num_frames=args.frames,
                input_file=args.input_file,
                cookies=args.cookies,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Pipeline error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
