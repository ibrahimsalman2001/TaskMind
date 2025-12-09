import os
import tempfile
import cv2
import subprocess
from pathlib import Path
from typing import List
import math
import concurrent.futures

try:
    import requests
except Exception:
    requests = None


def _concatenate_parts(parts, output_path: Path):
    with open(output_path, "wb") as out_f:
        for p in parts:
            with open(p, "rb") as in_f:
                out_f.write(in_f.read())


def _download_range(url: str, start: int, end: int, part_path: Path, headers=None):
    h = dict(headers or {})
    h["Range"] = f"bytes={start}-{end}"
    resp = requests.get(url, headers=h, stream=True, timeout=60)
    resp.raise_for_status()
    with open(part_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _ranged_download(url: str, output_path: Path, num_chunks: int = 8) -> bool:
    """Download `url` using HTTP Range requests in `num_chunks` parallel pieces.

    Returns True on success, False on failure (e.g., server doesn't support ranges).
    """
    if requests is None:
        return False

    # Make a HEAD request to get content-length and check range support
    head = requests.head(url, allow_redirects=True, timeout=30)
    if head.status_code >= 400:
        return False

    total = head.headers.get("Content-Length")
    accept_ranges = head.headers.get("Accept-Ranges", "")
    if total is None or ("bytes" not in accept_ranges.lower() and int(total) > 50_000_000):
        # If no content-length or no range support and file is large, bail out
        # (for small files we can try a single-threaded GET)
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception:
            return False

    total = int(total)
    # Compute chunk boundaries
    chunk_size = math.ceil(total / num_chunks)
    temp_dir = output_path.parent
    part_paths = []
    tasks = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, num_chunks)) as exe:
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size - 1, total - 1)
            part_path = temp_dir / f"{output_path.name}.part{i}"
            part_paths.append(part_path)
            tasks.append(exe.submit(_download_range, url, start, end, part_path))

        # wait for completion and raise if any failed
        for t in concurrent.futures.as_completed(tasks):
            try:
                t.result()
            except Exception:
                # cleanup
                for p in part_paths:
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass
                return False

    # Concatenate parts
    _concatenate_parts(part_paths, output_path)

    # Remove part files
    for p in part_paths:
        try:
            p.unlink()
        except Exception:
            pass

    return True


def download_video(video_url: str, output_dir: str, use_ranged: bool = True, cookies_path: str | None = None) -> str:
    """
    Download a YouTube video. Prefer parallel ranged download of the direct media URL (faster),
    with fallback to calling the `yt-dlp` subprocess to download normally.
    Returns path to local file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete old video files to force fresh download
    for ext in ("mp4", "mkv", "webm"):
        old_file = output_dir / f"video.{ext}"
        if old_file.exists():
            old_file.unlink()

    output_path = output_dir / "video.mp4"

    # Try to use yt_dlp Python API to get direct URL
    try:
        import yt_dlp
        ydl_opts = {"skip_download": True, "quiet": True}
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        # choose a suitable format with a direct URL (prefer mp4)
        formats = info.get("formats") or []
        direct_url = None
        # prefer mp4 with both video+audio
        for fmt in reversed(formats):
            if fmt.get("ext") == "mp4" and fmt.get("acodec") != "none":
                direct_url = fmt.get("url")
                break
        if direct_url is None and formats:
            # fallback to best format
            direct_url = formats[-1].get("url")

        if direct_url and use_ranged:
            # Attempt ranged download
            ok = _ranged_download(direct_url, output_path, num_chunks=8)
            if ok and output_path.exists():
                return str(output_path)
    except Exception:
        # fall back to subprocess method below
        pass

    # Last-resort: call yt-dlp subprocess to download directly to output_path
    try:
        output_template = str(output_dir / "video.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", "mp4",
            "-o", output_template,
            video_url,
        ]
        if cookies_path:
            cmd.insert(-1, "--cookies")
            cmd.insert(-1, cookies_path)
        subprocess.run(cmd, check=True)

        # locate file
        for ext in ("mp4", "mkv", "webm"):
            candidate = output_dir / f"video.{ext}"
            if candidate.exists():
                # if not mp4, rename to video.mp4
                if candidate.suffix != ".mp4":
                    candidate.rename(output_path)
                    return str(output_path)
                return str(candidate)
    except Exception as e:
        raise RuntimeError(f"Failed to download video via yt-dlp: {e}")

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


from vision_model import classify_frame

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=47fLXANW39k"
    frames = download_and_extract_frames(test_url, num_frames=6)

    print("\nRunning EfficientNet classification:")
    for frame in frames:
        print(f"\n[Frame: {frame}]")
        predictions = classify_frame(frame, top_k=3)
        for label, score in predictions:
            print(f"  → {label}: {score:.3f}")
