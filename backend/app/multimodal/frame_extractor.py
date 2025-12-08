import os
import tempfile
import cv2
import subprocess
from pathlib import Path
from typing import List


def download_video(video_url: str, output_dir: str) -> str:
    """
    Download a YouTube video using yt-dlp and return the local file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a fixed filename pattern
    output_template = str(output_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", output_template,
        video_url,
    ]

    subprocess.run(cmd, check=True)

    # Find the downloaded file (video.mp4 or similar)
    for ext in ("mp4", "mkv", "webm"):
        candidate = output_dir / f"video.{ext}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError("Could not find downloaded video file.")


def extract_sample_frames(
    video_path: str,
    num_frames: int = 8,
    output_dir: str | None = None,
) -> List[str]:
    """
    Extract `num_frames` evenly spaced frames from the video and save as images.
    Returns list of file paths to the extracted frames.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="taskmind_frames_")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise RuntimeError("Video has zero frames.")

    # indices of frames we want to sample
    step = max(frame_count // num_frames, 1)
    frame_indices = [i * step for i in range(num_frames)]
    frame_paths: List[str] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        frame_filename = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        frame_paths.append(str(frame_filename))

    cap.release()
    return frame_paths


def download_and_extract_frames(
    video_url: str,
    num_frames: int = 8,
) -> List[str]:
    """
    High-level helper: download a YouTube video and extract sample frames.
    Returns paths to extracted frames.
    """
    with tempfile.TemporaryDirectory(prefix="taskmind_video_") as tmp_dir:
        video_path = download_video(video_url, tmp_dir)
        frame_paths = extract_sample_frames(video_path, num_frames=num_frames)
        return frame_paths


if __name__ == "__main__":

    # quick manual test
    test_url = "https://www.youtube.com/watch?v=47fLXANW39k"
    frames = download_and_extract_frames(test_url, num_frames=6)
    print("Extracted frames:")
    for p in frames:
        print(p)