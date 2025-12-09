from frame_extractor import extract_sample_frames, download_video
from cv_classifier import classify_video_by_cv

# 1. Download video
video_url = "https://www.youtube.com/shorts/Ldx0J47Z65g"
video_path = download_video(video_url, "temp_videos")

# 2. Extract frames
frames = extract_sample_frames(video_path, num_frames=5)

# 3. Classify using CV module
scores = classify_video_by_cv(frames)

# 4. Output
for cat, score in sorted(scores.items(), key=lambda x: -x[1]):
    if score > 0:
        print(f"{cat}: {score}")

# from keyword_classifier import keyword_based_classifier

# ocr_text = "This bayan talks about the life of Prophet Muhammad and his teachings"
# metadata = "A short Islamic lecture on seerah for youth and beginners"

# combined_text = ocr_text + " " + metadata

# category, matches = keyword_based_classifier(combined_text)
# print(f"Category: {category} | Matches: {matches}")
