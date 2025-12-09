# test.py

from frame_extractor import download_and_extract_frames
from cv_classifier import classify_video_by_cv

# STEP 1: Provide your test video URL
video_url = "https://www.youtube.com/shorts/Ldx0J47Z65g"

# STEP 2: Download and extract frames from video
frames = download_and_extract_frames(video_url, num_frames=6)

# STEP 3: Classify using the full CV module
cv_scores = classify_video_by_cv(frames)

# STEP 4: Print the similarity score vector
print("\n🎯 CV Module Category Scores:")
for cat, score in sorted(cv_scores.items(), key=lambda x: -x[1]):
    if score > 0:
        print(f"{cat:<25} → {score}")